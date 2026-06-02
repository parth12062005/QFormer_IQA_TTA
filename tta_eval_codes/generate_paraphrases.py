"""
Pre-generate paraphrased prompts for all unique prompts in A20K test set using Groq API.
Batches 10 prompts per request to minimize API calls.
Saves to paraphrased_prompts.json.
"""
import os, json, time, requests
import pandas as pd

GROQ_KEY = "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"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.1-8b-instant"
OUTPUT_FILE = "paraphrased_prompts.json"
BATCH_SIZE = 5  # prompts per API call

_SPLIT_DIR = "../important split files-20260527T062853Z-3-001/important split files"
CSV_PATH = os.path.join(_SPLIT_DIR, "A20K_new", "A20k_test_full_PT1_normalized.csv")

def call_groq(prompts_batch):
    """Send a batch of prompts for paraphrasing."""
    numbered = "\n".join(f"{i+1}. {p}" for i, p in enumerate(prompts_batch))
    system_msg = (
        "You rephrase image generation prompts. For each numbered prompt, provide a slightly "
        "rephrased version that keeps the EXACT same visual meaning and semantics. "
        "Do NOT add or remove any visual elements. Only change wording slightly. "
        "Return ONLY the numbered rephrased prompts, one per line, matching the input numbering."
    )
    user_msg = f"Rephrase these prompts:\n{numbered}"
    
    resp = requests.post(GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": 2000,
            "temperature": 0.7,
        },
        timeout=30,
    )
    
    if resp.status_code == 429:
        # Rate limited — wait and retry
        wait = int(resp.headers.get("retry-after", 10))
        print(f"  Rate limited. Waiting {wait}s...")
        time.sleep(wait)
        return call_groq(prompts_batch)
    
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"].strip()
    
    # Parse numbered responses
    lines = [l.strip() for l in content.split("\n") if l.strip()]
    results = []
    for line in lines:
        # Remove numbering like "1. " or "1) "
        prefixes = []
        for i in range(len(prompts_batch)):
            prefixes.extend([f"{i+1}. ", f"{i+1}) ", f"{i+1}: "])
        for prefix in prefixes:
            if line.startswith(prefix):
                line = line[len(prefix):]
                break
        # Remove quotes
        line = line.strip('"').strip("'")
        results.append(line)
    
    return results


def main():
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()
    unique_prompts = df["prompt"].unique().tolist()
    
    print(f"Total unique prompts: {len(unique_prompts)}")
    
    # Load existing progress if any
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            mapping = json.load(f)
        print(f"Resuming from {len(mapping)} already paraphrased prompts")
    else:
        mapping = {}
    
    remaining = [p for p in unique_prompts if p not in mapping]
    print(f"Remaining to paraphrase: {len(remaining)}")
    
    for i in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[i:i+BATCH_SIZE]
        try:
            results = call_groq(batch)
            # Match results to originals
            for j, orig in enumerate(batch):
                if j < len(results) and results[j]:
                    mapping[orig] = results[j]
                else:
                    mapping[orig] = orig  # fallback to original
            
            # Save progress
            if (i // BATCH_SIZE) % 10 == 0:
                with open(OUTPUT_FILE, "w") as f:
                    json.dump(mapping, f, indent=2)
                    
            done = len(mapping)
            total = len(unique_prompts)
            print(f"  [{done}/{total}] ({100*done/total:.1f}%)")
            
            time.sleep(1.5)  # rate limit safety
            
        except Exception as e:
            print(f"  Error at batch {i}: {e}")
            time.sleep(5)
            continue
    
    # Final save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"\nDone! Saved {len(mapping)} paraphrased prompts to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
