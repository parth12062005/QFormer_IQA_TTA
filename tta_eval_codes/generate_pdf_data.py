import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from lavis.models import load_model_and_preprocess

# ── Paths ──
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
_SPLIT_DIR = os.path.join(_PROJECT_DIR, "important split files-20260527T062853Z-3-001", "important split files", "A20K_new")

TRAIN_CSV = os.path.join(_SPLIT_DIR, "A20k_train_full_PT1_normalized.csv")
VAL_CSV = os.path.join(_SPLIT_DIR, "A20k_val_full_PT1_normalized.csv")
TEST_CSV = os.path.join(_SPLIT_DIR, "A20k_test_full_PT1_normalized.csv")
EMBED_ROOT = "/media/parth/021f75bf-bae8-49ef-86a5-28ca19171835/parth/dataset/embeddings/a20k"
CHECKPOINT = os.path.join(_PROJECT_DIR, "checkpoints", "evalmi_baseline_qf.pth")
OUTPUT_CSV = "baseline_1000_sample.csv"

# ── Dataset ──
class QFormerEmbeddingDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row["image_name"])
        
        embed_path = os.path.join(EMBED_ROOT, img_name.replace(".png", ".npz").replace(".jpg", ".npz"))
        image_embeds = torch.from_numpy(np.load(embed_path)["embed"]).float()
        
        return {
            "image_embeds": image_embeds,
            "prompt": str(row["prompt"]),
            "description": str(row["gen_answer"]),
            "image_name": img_name,
            "gt_score": torch.tensor(float(row["gt_score"]), dtype=torch.float32),
        }

def collate_fn(batch):
    return {
        "image_embeds": torch.stack([b["image_embeds"] for b in batch], dim=0),
        "prompts": [b["prompt"] for b in batch],
        "descs": [b["description"] for b in batch],
        "image_names": [b["image_name"] for b in batch],
        "gt_scores": torch.stack([b["gt_score"] for b in batch], dim=0),
    }

# ── Models ──
class RegressorLinear(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.layer = nn.Linear(input_dim, output_dim)
    def forward(self, x): return self.layer(x)

class QformerWrapper(nn.Module):
    def __init__(self, device):
        super().__init__()
        model, _, _ = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)
        self.model = model.to(device)
        self.device = device
        del self.model.visual_encoder
        del self.model.ln_vision
        torch.cuda.empty_cache()

    def forward(self, image_embeds_frozen, prompts, descs):
        B = image_embeds_frozen.size(0)
        image_embeds_frozen = image_embeds_frozen.to(self.device)
        image_atts = torch.ones(image_embeds_frozen.size()[:-1], dtype=torch.long, device=self.device)
        query_tokens = self.model.query_tokens.expand(B, -1, -1)
        text_prompt = self.model.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=self.device)
        mm_attention_mask = torch.cat([query_atts, text_prompt.attention_mask], dim=1)
        
        mm_out = self.model.Qformer.bert(
            text_prompt.input_ids, query_embeds=query_tokens, attention_mask=mm_attention_mask,
            encoder_hidden_states=image_embeds_frozen, encoder_attention_mask=image_atts, return_dict=True
        )
        mm_mean_embeds = mm_out.last_hidden_state[:, :query_tokens.size(1), :].mean(dim=1)
        return mm_mean_embeds

def main():
    print("Loading data splits...")
    df_train = pd.read_csv(TRAIN_CSV)
    df_val = pd.read_csv(VAL_CSV)
    df_test = pd.read_csv(TEST_CSV)
    
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
    df_all.columns = df_all.columns.str.strip()
    
    print(f"Total images in A20K: {len(df_all)}")
    
    # Sample 1000
    df_sample = df_all.sample(n=1000, random_state=42).reset_index(drop=True)
    print(f"Sampled 1000 images.")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("Loading models...")
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    qformer = QformerWrapper(device)
    regressor = RegressorLinear(768, 1).to(device)
    
    qformer.model.Qformer.load_state_dict(ckpt["qformer.Qformer"], strict=True)
    qformer.model.query_tokens = nn.Parameter(ckpt["query_tokens"].to(device))
    regressor.load_state_dict(ckpt["regressor"], strict=True)
    
    qformer.eval()
    regressor.eval()
    
    dataset = QFormerEmbeddingDataset(df_sample)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            embeds = batch["image_embeds"].to(device)
            mm_embeds = qformer(embeds, batch["prompts"], batch["descs"])
            preds = regressor(mm_embeds).squeeze(-1).cpu().numpy()
            gts = batch["gt_scores"].numpy()
            
            for i in range(len(batch["image_names"])):
                rows.append({
                    "image_name": batch["image_names"][i],
                    "prompt": batch["prompts"][i],
                    "gen_answer": batch["descs"][i],
                    "gt_score": gts[i],
                    "pred_score": preds[i]
                })
                
    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved inference results to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
