"""
Experiment 2: CLIP-Based Quality Clustering Contrastive TTA

Instead of clustering by predicted score (GC), we use CLIP to compute
image similarity with "good quality" and "bad quality" text anchors.
Images closer to "good" → high cluster, closer to "bad" → low cluster.
Then apply contrastive loss on Q-Former projected features.

Config: LayerNorm, LR=5e-4, 3 steps, batch_size=8, A20K test set.
"""

import os, sys, random, json
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import clip

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_tta import (
    QformerWrapper, Regressor, ProjectionHead, DATASET_CONFIGS,
    DEFAULT_CHECKPOINT, set_seed, spearmanr, pearsonr, collate_fn,
)
from tta_framework.param_strategy import get_tta_params, freeze_all_except

# ── Config ─────────────────────────────────────────────────────────────────
DATASET    = "a20k"
SPLIT      = "test"
BATCH_SIZE = 8
TTA_STEPS  = 3
TTA_LR     = 5e-4
STRATEGY   = "layernorm"
GC_TEMP    = 0.5
GC_P       = 0.25
OUT_DIR    = "results/exp2_clip_cluster"

# CLIP quality anchor texts
GOOD_PROMPTS = [
    "A high quality, aesthetically pleasing, well-composed, beautiful image",
    "A stunning, detailed, professional quality photograph with excellent composition",
    "A visually appealing, sharp, well-lit, high-resolution image",
]
BAD_PROMPTS = [
    "A low quality, poorly generated, distorted, ugly, blurry image",
    "A terrible, artifact-ridden, deformed, low-resolution image with visual errors",
    "An unpleasant, badly composed, noisy, poorly generated image",
]

# ── Dataset that loads raw images for CLIP ─────────────────────────────────
class CLIPAugDataset(Dataset):
    def __init__(self, csv_path, cfg, clip_preprocess):
        self.df = pd.read_csv(csv_path)
        self.df.columns = self.df.columns.str.strip()
        self.cfg = cfg
        self.clip_preprocess = clip_preprocess
        
        # Build image lookup
        self.img_lookup = {}
        img_root = cfg["img_root"]
        if cfg.get("img_subdir"):
            for sd in os.listdir(img_root):
                sp = os.path.join(img_root, sd)
                if not os.path.isdir(sp): continue
                for f in os.listdir(sp):
                    if f.lower().endswith(('.png','.jpg','.jpeg')):
                        self.img_lookup[f] = os.path.join(sp, f)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row[self.cfg["img_col"]])
        
        # Load precomputed embedding
        base = img_name.replace(".png", self.cfg["embed_ext"]).replace(".jpg", self.cfg["embed_ext"])
        embed_path = os.path.join(self.cfg["embed_root"], base)
        if self.cfg["embed_load"] == "npy":
            image_embeds = torch.from_numpy(np.load(embed_path)).float()
        else:
            image_embeds = torch.from_numpy(np.load(embed_path)["embed"]).float()
        
        # Load raw image for CLIP
        img_path = self.img_lookup.get(img_name)
        clip_image = self.clip_preprocess(Image.open(img_path).convert("RGB"))
        
        prompt = str(row.get(self.cfg.get("prompt_col", "prompt"), ""))
        desc = str(row.get(self.cfg.get("desc_col", "gen_answer"), ""))
        gt = torch.tensor(float(row[self.cfg["gt_col"]]), dtype=torch.float32)
        
        return {
            "image_embeds": image_embeds,
            "clip_image": clip_image,
            "prompt": prompt,
            "description": desc,
            "image_name": img_name,
            "gt_score": gt,
        }

def clip_collate_fn(batch):
    return {
        "image_embeds": torch.stack([b["image_embeds"] for b in batch]),
        "clip_images": torch.stack([b["clip_image"] for b in batch]),
        "prompts": [b["prompt"] for b in batch],
        "descs": [b["description"] for b in batch],
        "image_names": [b["image_name"] for b in batch],
        "gt_scores": torch.stack([b["gt_score"] for b in batch]),
    }


def clip_quality_scores(clip_model, clip_images, good_text_feats, bad_text_feats, device):
    """Compute per-image quality signal using CLIP similarity."""
    with torch.no_grad():
        image_feats = clip_model.encode_image(clip_images.to(device))
        image_feats = F.normalize(image_feats.float(), dim=-1)
        
        # Average similarity with good and bad anchors
        sim_good = (image_feats @ good_text_feats.T).mean(dim=-1)
        sim_bad  = (image_feats @ bad_text_feats.T).mean(dim=-1)
        
        # Quality signal: how much more "good" than "bad"
        quality = sim_good - sim_bad
    return quality


def clip_contrastive_loss(proj_feats, quality_signal, p=0.25, temperature=0.5, device=None):
    """GC-style contrastive loss but using CLIP quality signal for clustering."""
    B = proj_feats.size(0)
    k = max(2, int(B * p))
    
    if B < 4:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Cluster by CLIP quality signal
    idx = torch.argsort(quality_signal)
    emb_low  = proj_feats[idx[:k]]
    emb_high = proj_feats[idx[-k:]]
    n_i, n_j = emb_low.size(0), emb_high.size(0)
    
    z_i = F.normalize(emb_low, dim=1)
    z_j = F.normalize(emb_high, dim=1)
    
    representations = torch.cat([z_i, z_j], dim=0)
    sim_matrix = F.cosine_similarity(
        representations.unsqueeze(1), representations.unsqueeze(0), dim=2
    )
    
    pos_sim_ij = sim_matrix[:n_i, :n_j]
    mask_ij = (~torch.eye(n_i, n_j, dtype=bool, device=device)).float()
    sim_ij = torch.sum(pos_sim_ij * mask_ij, dim=1) / max(n_j - 1, 1)
    
    pos_sim_ji = sim_matrix[n_i:, n_j:]
    mask_ji = (~torch.eye(n_j, n_i, dtype=bool, device=device)).float()
    sim_ji = torch.sum(pos_sim_ji * mask_ji, dim=1) / max(n_i - 1, 1)
    
    positives = torch.cat([sim_ij, sim_ji], dim=0)
    
    total = n_i + n_j
    neg_mask = torch.ones(total, total, dtype=bool, device=device)
    neg_mask[:n_i, :n_j] = False
    neg_mask[n_i:, n_j:] = False
    neg_mask = neg_mask.float()
    
    nom = torch.exp(positives / temperature)
    denom = neg_mask * torch.exp(sim_matrix / temperature)
    
    loss_partial = torch.sum(nom / (nom + torch.sum(denom, dim=1) + 1e-8)) / total
    loss = -torch.log(loss_partial + 1e-8)
    return loss


def main():
    set_seed(1234)
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    cfg = DATASET_CONFIGS[DATASET]
    csv_path = cfg["splits"][SPLIT]
    
    # ── Load CLIP model ──
    print("Loading CLIP ViT-B/32...")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    
    # Pre-encode quality anchor texts
    good_tokens = clip.tokenize(GOOD_PROMPTS).to(device)
    bad_tokens  = clip.tokenize(BAD_PROMPTS).to(device)
    with torch.no_grad():
        good_text_feats = F.normalize(clip_model.encode_text(good_tokens).float(), dim=-1)
        bad_text_feats  = F.normalize(clip_model.encode_text(bad_tokens).float(), dim=-1)
    
    # ── Load dataset ──
    print(f"Loading dataset: {DATASET} / {SPLIT}")
    ds = CLIPAugDataset(csv_path, cfg, clip_preprocess)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=clip_collate_fn, num_workers=4, pin_memory=True)
    
    # ── Load Q-Former ──
    print(f"Loading Q-Former checkpoint...")
    ckpt = torch.load(DEFAULT_CHECKPOINT, map_location="cpu", weights_only=False)
    qformer   = QformerWrapper(device, is_eval=False).to(device)
    regressor = Regressor(input_dim=768, output_dim=1).to(device)
    proj_head = ProjectionHead(input_dim=768, output_dim=128).to(device)
    
    qformer.model.Qformer.load_state_dict(ckpt["qformer.Qformer"], strict=True)
    qformer.model.query_tokens = nn.Parameter(ckpt["query_tokens"].to(device))
    regressor.load_state_dict(ckpt["regressor"], strict=True)
    
    # Snapshot for reset
    orig_qt = qformer.model.query_tokens.detach().clone()
    ln_mods = [m for m in qformer.model.Qformer.modules() if isinstance(m, nn.LayerNorm)]
    orig_ln = [{"w": m.weight.detach().clone(), "b": m.bias.detach().clone()} for m in ln_mods]
    
    print(f"\nRunning CLIP-Cluster TTA: strategy={STRATEGY}, lr={TTA_LR}, steps={TTA_STEPS}, bs={BATCH_SIZE}")
    print(f"Total batches: {len(loader)}\n")
    
    all_rows = []
    
    for batch in tqdm(loader, desc="CLIP-Cluster TTA"):
        image_embeds = batch["image_embeds"].to(device)
        clip_images  = batch["clip_images"]
        prompts = batch["prompts"]
        descs   = batch["descs"]
        gt_scores = batch["gt_scores"].numpy()
        names = batch["image_names"]
        B = image_embeds.size(0)
        
        # Reset model
        with torch.no_grad():
            qformer.model.query_tokens.copy_(orig_qt)
            for m, s in zip(ln_mods, orig_ln):
                m.weight.copy_(s["w"]); m.bias.copy_(s["b"])
        
        # Baseline prediction
        qformer.eval(); regressor.eval()
        with torch.no_grad():
            mm = qformer.forward_qformer(image_embeds, prompts, descs)
            baseline_preds = regressor(mm).squeeze(-1).cpu().numpy()
        
        batch_rows = []
        for i in range(B):
            batch_rows.append({
                "image_name": names[i], "gt_score": gt_scores[i],
                "pred_baseline": baseline_preds[i],
            })
        
        # Compute CLIP quality signal (fixed, not updated during TTA)
        quality_signal = clip_quality_scores(
            clip_model, clip_images, good_text_feats, bad_text_feats, device
        )
        
        # Setup TTA
        for p in qformer.model.parameters(): p.requires_grad = False
        for p in regressor.parameters(): p.requires_grad = False
        for p in proj_head.parameters(): p.requires_grad = False
        
        ln_params = get_tta_params(qformer, STRATEGY)
        freeze_all_except(qformer, ln_params)
        optimizer = torch.optim.Adam(ln_params, lr=TTA_LR)
        qformer.train()
        
        for step in range(1, TTA_STEPS + 1):
            optimizer.zero_grad()
            
            mm = qformer.forward_qformer(image_embeds, prompts, descs)
            proj_feats = proj_head(mm)
            
            loss = clip_contrastive_loss(
                proj_feats, quality_signal,
                p=GC_P, temperature=GC_TEMP, device=device
            )
            
            if loss.item() > 0:
                loss.backward()
                optimizer.step()
            
            # Record predictions
            qformer.eval()
            with torch.no_grad():
                mm_post = qformer.forward_qformer(image_embeds, prompts, descs)
                step_preds = regressor(mm_post).squeeze(-1).cpu().numpy()
            qformer.train()
            
            for i in range(B):
                batch_rows[i][f"pred_step{step}"] = step_preds[i]
        
        all_rows.extend(batch_rows)
    
    # ── Save results ──
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(OUT_DIR, "full_results.csv"), index=False)
    
    stages = ["pred_baseline", "pred_step1", "pred_step2", "pred_step3"]
    labels = ["Baseline", "Step 1", "Step 2", "Step 3"]
    metrics = []
    for s, l in zip(stages, labels):
        srcc = spearmanr(df[s].values, df["gt_score"].values)
        plcc = pearsonr(df[s].values, df["gt_score"].values)
        metrics.append({"stage": l, "srcc": srcc, "plcc": plcc})
        print(f"  {l:>10s}: SRCC={srcc:.6f}  PLCC={plcc:.6f}")
    
    pd.DataFrame(metrics).to_csv(os.path.join(OUT_DIR, "metrics.csv"), index=False)
    
    # ── Plots ──
    srcc_vals = [m["srcc"] for m in metrics]
    plcc_vals = [m["plcc"] for m in metrics]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    ax.plot(x, srcc_vals, "o-", color="#2563EB", linewidth=2.5, markersize=10, label="SRCC")
    ax.plot(x, plcc_vals, "s--", color="#DC2626", linewidth=2.5, markersize=10, label="PLCC")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Correlation", fontsize=13)
    ax.set_title("Exp 2: CLIP-Cluster Contrastive TTA (A20K)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12); ax.grid(True, alpha=0.3)
    for i, (s, p) in enumerate(zip(srcc_vals, plcc_vals)):
        ax.annotate(f"{s:.4f}", (x[i], s), textcoords="offset points", xytext=(0, 12), fontsize=10, color="#2563EB", ha="center")
        ax.annotate(f"{p:.4f}", (x[i], p), textcoords="offset points", xytext=(0, -18), fontsize=10, color="#DC2626", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "metric_progression.png"), dpi=200)
    plt.close()
    
    # Scatter
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, stage, label in zip(axes, ["pred_baseline", "pred_step3"], ["Baseline", "After 3 TTA Steps"]):
        ax.scatter(df["gt_score"], df[stage], alpha=0.15, s=8, color="#6366F1")
        lims = [min(df["gt_score"].min(), df[stage].min())-0.05, max(df["gt_score"].max(), df[stage].max())+0.05]
        ax.plot(lims, lims, "r--", alpha=0.6, label="Ideal (y=x)")
        s = spearmanr(df[stage].values, df["gt_score"].values)
        p = pearsonr(df[stage].values, df["gt_score"].values)
        ax.set_title(f"{label}\nSRCC={s:.4f}  PLCC={p:.4f}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Ground Truth Score", fontsize=11); ax.set_ylabel("Predicted Score", fontsize=11)
        ax.legend(fontsize=10); ax.grid(True, alpha=0.2)
    plt.suptitle("Exp 2: CLIP-Cluster Contrastive TTA", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "scatter.png"), dpi=200, bbox_inches="tight")
    plt.close()
    
    # Error histogram
    df["err_bl"] = np.abs(df["pred_baseline"] - df["gt_score"])
    df["err_s3"] = np.abs(df["pred_step3"] - df["gt_score"])
    df["err_chg"] = df["err_s3"] - df["err_bl"]
    improved = (df["err_chg"] < 0).sum()
    worsened = (df["err_chg"] > 0).sum()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["err_chg"], bins=80, color="#8B5CF6", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.axvline(df["err_chg"].mean(), color="#059669", linestyle="-", linewidth=2, label=f"Mean Δ={df['err_chg'].mean():.4f}")
    ax.set_title(f"Per-Image Error Change (CLIP-Cluster TTA)\nImproved: {improved} | Worsened: {worsened}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Error Change (neg=improved)"); ax.set_ylabel("Count")
    ax.legend(); ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "error_histogram.png"), dpi=200)
    plt.close()
    
    print(f"\nAll results saved to {OUT_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
