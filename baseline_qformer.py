##### Script is to pretrain the baseline qformer with EvalMI 
##
#### script for FULL LABEL with baseline qformer where we only use mm pass only 
#### - Supervised: regression (multimodal query tokens pooled -> MLP -> MOS)
#### We save the predictions for test set only here .
#### NOTE: Uses PRECOMPUTED ViT embeddings. Run precompute_embeddings.py first.

##### ----------------- ####
#####  1) IMPORTS + UTILS
##### ----------------- ####
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from lavis.models import load_model_and_preprocess

##### SET SEED #####
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 1234
set_seed(SEED)

#### METRICS ####
# ----------------------------
# NumPy-only implementation
# ----------------------------
def rankdata_numpy(a):
    """
    NumPy-only equivalent of scipy.stats.rankdata(method="average")
    """
    a = np.asarray(a)
    sorter = np.argsort(a)
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(a))

    a_sorted = a[sorter]
    obs = np.concatenate(([True], a_sorted[1:] != a_sorted[:-1]))
    dense_rank = np.cumsum(obs)

    counts = np.bincount(dense_rank)
    cumulative = np.cumsum(counts)

    ranks = (cumulative[dense_rank] + cumulative[dense_rank - 1] + 1) / 2.0
    return ranks[inv]


def spearmanr_numpy(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    assert x.shape == y.shape, "x and y must have same shape"

    rx = rankdata_numpy(x)
    ry = rankdata_numpy(y)

    rx = rx - rx.mean()
    ry = ry - ry.mean()

    denom = np.sqrt(np.sum(rx**2) * np.sum(ry**2))
    if denom == 0:
        return np.nan
    return np.sum(rx * ry) / denom


##### ----------------- ####
#####  2) DATASET + COLLATE (uses precomputed ViT embeddings)
##### ----------------- ####
IMG_COL    = "image_name"
PROMPT_COL = "prompt"
DESC_COL   = "gen_answer"   # change if your csv uses a different name
GT_COL     = "gt_score"

EMBED_ROOT = "../EvalMi-50K/embeddings"  # directory with precomputed .pt files

class QFormerEmbeddingDataset(Dataset):
    """Loads precomputed ViT embeddings instead of raw images."""
    def __init__(self, csv_path, embed_root=EMBED_ROOT):
        self.df = pd.read_csv(csv_path)
        self.embed_root = embed_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row[IMG_COL])

        # Load precomputed embedding: e.g. "ali_flux_schnell/001.png" -> "ali_flux_schnell/001.pt"
        embed_path = os.path.join(
            self.embed_root,
            img_name.replace(".png", ".pt").replace(".jpg", ".pt"),
        )
        image_embeds = torch.load(embed_path, weights_only=True).float()  # (num_patches, embed_dim)

        prompt = str(row[PROMPT_COL])
        desc   = str(row[DESC_COL])
        gt     = torch.tensor(float(row[GT_COL]), dtype=torch.float32)

        return {
            "image_embeds": image_embeds,
            "prompt": prompt,
            "description": desc,
            "image_name": img_name,
            "gt_score": gt,
        }

def collate_fn(batch):
    image_embeds = torch.stack([b["image_embeds"] for b in batch], dim=0)
    prompts = [b["prompt"] for b in batch]
    descs = [b["description"] for b in batch]
    image_names = [b["image_name"] for b in batch]
    gt_scores = torch.stack([b["gt_score"] for b in batch], dim=0)
    return {
        "image_embeds": image_embeds,  # (B, num_patches, embed_dim)
        "prompts": prompts,            # list[str]
        "descs": descs,                # list[str]
        "image_names": image_names,    # list[str]
        "gt_scores": gt_scores,        # (B,)
    }


##### ----------------- ####
#####  3) DEVICE
##### ----------------- ####
# Use GPU 1 (GPU 0 runs the display server)
device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
print(f"device: {device}")


##### ----------------- ####
#####  4) 
##### ----------MODELS------- ####
# class Regressor(nn.Module):
#     def __init__(self, input_dim, output_dim=1):
#         super().__init__()
#         self.layer = nn.Linear(input_dim, output_dim)

#     def forward(self, x):
#         return self.layer(x)

class Regressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layer(x)

class QformerWrapper(nn.Module):
    """
    Q-Former wrapper that takes PRECOMPUTED image embeddings instead of raw images.
    No ViT encoder needed — only loads the Q-Former + query tokens.
    """
    def __init__(self, device, is_eval=False):
        super().__init__()
        model, vis_proc, txt_proc = load_model_and_preprocess(
            name="blip2_feature_extractor",
            model_type="pretrain",
            is_eval=is_eval,
            device=device,
        )
        self.model = model.to(device)
        self.device = device

        # Free ViT from GPU memory since we use precomputed embeddings
        del self.model.visual_encoder
        del self.model.ln_vision
        torch.cuda.empty_cache()

    def forward(self, image_embeds_frozen, prompts, descs):
        """
        Takes precomputed frozen image embeddings and runs only the Q-Former.

        Args:
            image_embeds_frozen: (B, num_patches, embed_dim) - precomputed ViT output
            prompts:             list[str]
            descs:               list[str] (unused in baseline, kept for API compat)

        Returns:
            mm_mean_embeds:      (B, 768)
        """
        B = image_embeds_frozen.size(0)
        image_embeds_frozen = image_embeds_frozen.to(self.device)

        image_atts = torch.ones(
            image_embeds_frozen.size()[:-1], dtype=torch.long, device=self.device
        )

        query_tokens = self.model.query_tokens.expand(B, -1, -1)  # [B,32,768]
        # -------------------------
        # MULTIMODAL PASS (PROMPTS)
        # -------------------------
        text_prompt = self.model.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=self.device)
        mm_attention_mask = torch.cat([query_atts, text_prompt.attention_mask], dim=1)

        mm_out = self.model.Qformer.bert(
            text_prompt.input_ids,
            query_embeds=query_tokens,
            attention_mask=mm_attention_mask,
            encoder_hidden_states=image_embeds_frozen,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        mm_query_embeds = mm_out.last_hidden_state[:, : query_tokens.size(1), :]  # [B,32,768]
        mm_mean_embeds = mm_query_embeds.mean(dim=1)                               # [B,768]

        return mm_mean_embeds


##### ----------------- ####
#####  5) INIT + FREEZE/UNFREEZE
##### ----------------- ####
qformer = QformerWrapper(device=device, is_eval=False).to(device)
regressor = Regressor(input_dim=768, hidden_dim=256, output_dim=1).to(device)

# Freeze everything in BLIP2 feature extractor
for p in qformer.model.parameters():
    p.requires_grad = False

# Unfreeze ONLY query tokens + Qformer (as in your rough script intent)
qformer.model.query_tokens.requires_grad = True
for p in qformer.model.Qformer.parameters():
    p.requires_grad = True

for p in regressor.parameters():
    p.requires_grad = True

##### ----------------- ####
#####  6) LOSSES + OPTIM
##### ----------------- ####
reg_criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    [p for p in qformer.model.parameters() if p.requires_grad] + list(regressor.parameters()),
    lr=1e-4
)


##### ----------------- ####
#####  7) TRAIN / EVAL
##### ----------------- ####
def train_one_epoch(dataloader):
    qformer.train()
    regressor.train()

    total_loss = 0.0
    for batch in tqdm(dataloader, total=len(dataloader), desc="Training qformer"):
        image_embeds = batch["image_embeds"].to(device)
        prompts = batch["prompts"]
        descs = batch["descs"]
        gt_scores = batch["gt_scores"].to(device)  # (B,)

        optimizer.zero_grad()

        mm_mean_embeds = qformer(image_embeds, prompts, descs)

        pred = regressor(mm_mean_embeds).squeeze(-1)  # (B,)
        reg_loss = reg_criterion(pred, gt_scores)

        loss = reg_loss
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().item())

    return total_loss / max(1, len(dataloader))


@torch.no_grad()
def evaluate_and_save_df(dataloader, output_csv_path=None, desc_tag="Evaluating"):
    """
    Runs inference, computes SRCC, and returns a pandas DataFrame with per-sample records:
      image_name, prompt, gen_answer, gt_score, pred_score

    If output_csv_path is provided, saves the DataFrame to CSV.
    """
    qformer.eval()
    regressor.eval()

    rows = []
    pred_list = []
    gt_list = []

    for batch in tqdm(dataloader, total=len(dataloader), desc=desc_tag):
        image_embeds = batch["image_embeds"].to(device, non_blocking=True)
        image_names = batch["image_names"]
        prompts = batch["prompts"]
        descs = batch["descs"]
        gt_scores = batch["gt_scores"].to(device, non_blocking=True)  # (B,)

        mm_mean_embeds = qformer(image_embeds, prompts, descs)
        pred = regressor(mm_mean_embeds).squeeze(-1)  # (B,)

        # accumulate for SRCC
        pred_list.append(pred.detach().float().cpu())
        gt_list.append(gt_scores.detach().float().cpu())

        # store per-sample rows for later analysis
        pred_cpu = pred.detach().float().cpu().numpy()
        gt_cpu   = gt_scores.detach().float().cpu().numpy()

        for i in range(len(image_names)):
            rows.append({
                "image_name": image_names[i],
                "prompt": prompts[i],
                "gen_answer": descs[i],
                "gt_score": float(gt_cpu[i]),
                "pred_score": float(pred_cpu[i]),
            })

    pred_scores = torch.cat(pred_list, dim=0).numpy()
    gt_scores   = torch.cat(gt_list, dim=0).numpy()

    srcc = spearmanr_numpy(pred_scores, gt_scores)

    df = pd.DataFrame(rows)

    if output_csv_path is not None:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        print(f"[Saved] Per-sample results CSV: {output_csv_path}")

    return srcc, df


##### ----------------- ####
#####  8) MAIN
##### ----------------- ####
def main():
    print("PRETRAINING OF BASELINE QFORMER ON EVALMI DATABASE (using precomputed embeddings)")
    train_dataset = QFormerEmbeddingDataset(
        csv_path="../EvalMi-50K/evalmi_train.csv",
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )

    val_dataset = QFormerEmbeddingDataset(
        csv_path="../EvalMi-50K/evalmi_val.csv",
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )

    test_dataset = QFormerEmbeddingDataset(
        csv_path="../EvalMi-50K/evalmi_test.csv",
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )

    checkpoint_path = "./checkpoints/evalmi_baseline_qf_ver2.pth"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    best_srcc, best_test_srcc = -1.0, -1.0
    test_out_csv = "./results/evalmi_baseline_qf_test_ver2.csv"

    num_epochs = 15
    for epoch in tqdm(range(num_epochs), desc="Epochs", total=num_epochs):
        train_loss = train_one_epoch(train_dataloader)
        val_srcc,_= evaluate_and_save_df(val_dataloader, output_csv_path=None, desc_tag="Evaluating on evalmi validation")

        print(f"\nepoch-- {epoch}")
        print(f"train loss-- {train_loss:.6f}")
        print(f"val evaluated, Val srcc-- {val_srcc:.6f}")

        if val_srcc > best_srcc:
            best_srcc = val_srcc
            test_srcc, test_df= evaluate_and_save_df(test_dataloader, output_csv_path=test_out_csv, desc_tag="Evaluating on evalmi test")
            best_test_srcc = test_srcc
 
            print("test evaluated ,Test srcc--", test_srcc)

            torch.save(
                {
                    "qformer.Qformer": qformer.model.Qformer.state_dict(),
                    "query_tokens": qformer.model.query_tokens.detach().cpu(),
                    "regressor": regressor.state_dict(),
                },
                checkpoint_path,
            )
            print("model saved")

        print("best val srcc--", best_srcc)
        print("best test srcc--", best_test_srcc)


if __name__ == "__main__":
    main()