"""
Precompute VLM pseudo-scores for LLM-TTA clustering using Qwen2.5-VL-3B-Instruct.
This script generates scores based on the logits of 'good' and 'poor' tokens as 
described in the LLM-TTA paper.

Usage:
    python3 precompute_qwen_scores.py
"""

import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# --- CONFIG ---
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Image root contains train/, val/, test/ subdirs with flat image files
IMG_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "../../agiqa-20k/AIGCQA-30K-Image"))

# Split CSVs from the important split files folder (ONLY A20K TEST)
_SPLIT_DIR = os.path.join(
    _SCRIPT_DIR,
    "../important split files-20260527T062853Z-3-001",
    "important split files",
    "A20K_new",
)
CSV_PATHS = [
    os.path.join(_SPLIT_DIR, "A20k_test_full_PT1_normalized.csv")
]

OUT_JSON = "/media/parth/021f75bf-bae8-49ef-86a5-28ca19171835/parth/dataset/embeddings/a20k_qwen_scores.json"

# Token IDs for 'good' and 'poor' (including space prefixes)
# From tokenizer: " good" -> 1661, "good" -> 18536
# From tokenizer: " poor" -> 7852, "poor" -> 5368
GOOD_IDS = [1661, 18536, 3217, 10243] # added ' Good', 'Good' just in case
POOR_IDS = [7852, 5368, 14619, 15151] # added ' Poor', 'Poor' just in case

def _build_image_lookup(img_root):
    lookup = {}
    for subdir in os.listdir(img_root):
        subdir_path = os.path.join(img_root, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for fname in os.listdir(subdir_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                lookup[fname] = os.path.join(subdir_path, fname)
    return lookup

def main():
    print(f"Scanning image directory: {IMG_ROOT}")
    image_lookup = _build_image_lookup(IMG_ROOT)

    # Collect images
    all_image_names = set()
    for csv_path in CSV_PATHS:
        df = pd.read_csv(csv_path)
        all_image_names.update(df["image_name"].tolist())
    all_image_names = sorted(all_image_names)

    # Filter out already-computed scores
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON, "r") as f:
            scores_dict = json.load(f)
    else:
        scores_dict = {}

    remaining = [n for n in all_image_names if n not in scores_dict]
    print(f"Total test images: {len(all_image_names)}")
    print(f"Already computed: {len(scores_dict)}")
    print(f"Remaining: {len(remaining)}")

    if len(remaining) == 0:
        print("All scores computed. Exiting.")
        return

    # Load Model
    print(f"Loading {MODEL_ID}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # Inference Loop
    print("Computing pseudo-scores...")
    for name in tqdm(remaining):
        if name not in image_lookup:
            print(f"Warning: {name} not found on disk. Skipping.")
            continue
            
        img_path = image_lookup[name]
        
        # Prepare inputs
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path, "max_pixels": 262144}, # 512x512
                    {"type": "text", "text": "The quality of the image is"},
                ],
            }
        ]
        
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device=DEVICE)

        # Forward pass to get logits for the very next token
        with torch.no_grad():
            outputs = model(**inputs)
            
        # The logits for the last token in the input sequence
        next_token_logits = outputs.logits[0, -1, :]
        
        # Extract probabilities for 'good' and 'poor'
        good_logits = next_token_logits[GOOD_IDS]
        poor_logits = next_token_logits[POOR_IDS]
        
        # LogSumExp over the variations of "good" and "poor" to aggregate probabilities
        logit_good = torch.logsumexp(good_logits, dim=0)
        logit_poor = torch.logsumexp(poor_logits, dim=0)
        
        # Final Score: exp(good) / (exp(good) + exp(poor)) -> Sigmoid(good - poor)
        score = torch.sigmoid(logit_good - logit_poor).item()
        
        scores_dict[name] = score
        
        # Save incrementally
        if len(scores_dict) % 50 == 0:
            with open(OUT_JSON, "w") as f:
                json.dump(scores_dict, f, indent=4)

    # Final save
    with open(OUT_JSON, "w") as f:
        json.dump(scores_dict, f, indent=4)
        
    print(f"Done! Scores saved to {OUT_JSON}")

if __name__ == "__main__":
    main()
