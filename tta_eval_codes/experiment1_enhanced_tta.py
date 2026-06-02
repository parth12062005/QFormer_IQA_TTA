"""
Experiment 1: Enhanced TTA — GC + Novel Rank + View Consistency + Prompt Consistency

Losses:
  1. GC (existing) — cluster by predicted score
  2. Enhanced Rank — color jitter (weak) vs Fourier masking + noise (strong)
  3. View Consistency — variance of preds across flip/crop/original
  4. Prompt Consistency — MSE(pred(orig_prompt), pred(paraphrased_prompt))

Config: LayerNorm, LR=5e-4, 3 steps, batch_size=8, A20K test set.
Requires: raw images (keep_vit=True) and paraphrased_prompts.json
"""

import os, sys, json, random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_tta import (
    QformerWrapper, Regressor, ProjectionHead, DATASET_CONFIGS,
    DEFAULT_CHECKPOINT, set_seed, spearmanr, pearsonr,
)
from tta_framework.param_strategy import get_tta_params, freeze_all_except

# ── Config ─────────────────────────────────────────────────────────────────
DATASET    = "a20k"
SPLIT      = "test"
BATCH_SIZE = 4
TTA_STEPS  = 3
TTA_LR     = 5e-4
STRATEGY   = "layernorm"
GC_TEMP    = 0.5
GC_P       = 0.25
OUT_DIR    = "results/exp1_enhanced_tta"
PARAPHRASE_FILE = "paraphrased_prompts.json"

# Loss weights
W_GC       = 1.0
W_RANK     = 1.0
W_CONSIST  = 0.5
W_PROMPT   = 0.5

# ── Augmentations ──────────────────────────────────────────────────────────
# Weak: color/brightness jitter + mild crop
weak_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                         std=(0.26862954, 0.26130258, 0.27577711)),
])

# Base transform (for original)
base_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                         std=(0.26862954, 0.26130258, 0.27577711)),
])

# View consistency transforms
flip_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                         std=(0.26862954, 0.26130258, 0.27577711)),
])

crop_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 0.9)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                         std=(0.26862954, 0.26130258, 0.27577711)),
])


def fourier_strong_augment(img_tensor):
    """
    Fourier-domain augmentation:
    - Mask low-frequency amplitude (center 30% of FFT)
    - Inject high-frequency Gaussian noise
    """
    # img_tensor: (C, H, W)
    C, H, W = img_tensor.shape
    augmented = []
    for c in range(C):
        channel = img_tensor[c]
        # FFT
        fft = torch.fft.fft2(channel)
        fft_shifted = torch.fft.fftshift(fft)
        
        # Mask center 30% of amplitude
        amp = torch.abs(fft_shifted)
        phase = torch.angle(fft_shifted)
        
        cy, cx = H // 2, W // 2
        r = int(min(H, W) * 0.15)  # 30% diameter = 15% radius
        y_coords, x_coords = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        mask = ((y_coords - cy) ** 2 + (x_coords - cx) ** 2) > r ** 2
        amp_masked = amp * mask.float()
        
        # Reconstruct
        fft_modified = amp_masked * torch.exp(1j * phase)
        fft_unshifted = torch.fft.ifftshift(fft_modified)
        reconstructed = torch.fft.ifft2(fft_unshifted).real
        
        # Add high-freq noise
        noise = torch.randn_like(reconstructed) * 0.05
        reconstructed = reconstructed + noise
        
        augmented.append(reconstructed)
    
    return torch.stack(augmented)


# ── Dataset ────────────────────────────────────────────────────────────────
class EnhancedTTADataset(Dataset):
    def __init__(self, csv_path, cfg):
        self.df = pd.read_csv(csv_path)
        self.df.columns = self.df.columns.str.strip()
        self.cfg = cfg
        
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
        
        # Precomputed embedding
        base = img_name.replace(".png", self.cfg["embed_ext"]).replace(".jpg", self.cfg["embed_ext"])
        embed_path = os.path.join(self.cfg["embed_root"], base)
        if self.cfg["embed_load"] == "npy":
            image_embeds = torch.from_numpy(np.load(embed_path)).float()
        else:
            image_embeds = torch.from_numpy(np.load(embed_path)["embed"]).float()
        
        # Load raw image for augmentations
        img_path = self.img_lookup.get(img_name)
        pil_img = Image.open(img_path).convert("RGB")
        
        img_base   = base_transform(pil_img)
        img_weak   = weak_transform(pil_img)
        img_strong = fourier_strong_augment(base_transform(pil_img))
        img_flip   = flip_transform(pil_img)
        img_crop   = crop_transform(pil_img)
        
        prompt = str(row.get(self.cfg.get("prompt_col", "prompt"), ""))
        desc = str(row.get(self.cfg.get("desc_col", "gen_answer"), ""))
        gt = torch.tensor(float(row[self.cfg["gt_col"]]), dtype=torch.float32)
        
        return {
            "image_embeds": image_embeds,
            "img_base": img_base,
            "img_weak": img_weak,
            "img_strong": img_strong,
            "img_flip": img_flip,
            "img_crop": img_crop,
            "prompt": prompt,
            "description": desc,
            "image_name": img_name,
            "gt_score": gt,
        }


def enhanced_collate_fn(batch):
    return {
        "image_embeds": torch.stack([b["image_embeds"] for b in batch]),
        "img_base":   torch.stack([b["img_base"] for b in batch]),
        "img_weak":   torch.stack([b["img_weak"] for b in batch]),
        "img_strong": torch.stack([b["img_strong"] for b in batch]),
        "img_flip":   torch.stack([b["img_flip"] for b in batch]),
        "img_crop":   torch.stack([b["img_crop"] for b in batch]),
        "prompts": [b["prompt"] for b in batch],
        "descs": [b["description"] for b in batch],
        "image_names": [b["image_name"] for b in batch],
        "gt_scores": torch.stack([b["gt_score"] for b in batch]),
    }


# ── Losses ─────────────────────────────────────────────────────────────────
def gc_loss(proj_feats, preds, p=0.25, temperature=0.5, device=None):
    """Standard GC loss."""
    B = proj_feats.size(0)
    k = max(2, int(B * p))
    if B < 4:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    idx = torch.argsort(preds)
    z_i = F.normalize(proj_feats[idx[:k]], dim=1)
    z_j = F.normalize(proj_feats[idx[-k:]], dim=1)
    n_i, n_j = z_i.size(0), z_j.size(0)
    
    reps = torch.cat([z_i, z_j], dim=0)
    sim = F.cosine_similarity(reps.unsqueeze(1), reps.unsqueeze(0), dim=2)
    
    ps_ij = sim[:n_i, :n_j]
    m_ij = (~torch.eye(n_i, n_j, dtype=bool, device=device)).float()
    s_ij = torch.sum(ps_ij * m_ij, dim=1) / max(n_j - 1, 1)
    
    ps_ji = sim[n_i:, n_j:]
    m_ji = (~torch.eye(n_j, n_i, dtype=bool, device=device)).float()
    s_ji = torch.sum(ps_ji * m_ji, dim=1) / max(n_i - 1, 1)
    
    pos = torch.cat([s_ij, s_ji], dim=0)
    total = n_i + n_j
    nm = torch.ones(total, total, dtype=bool, device=device)
    nm[:n_i, :n_j] = False; nm[n_i:, n_j:] = False
    nm = nm.float()
    
    nom = torch.exp(pos / temperature)
    den = nm * torch.exp(sim / temperature)
    lp = torch.sum(nom / (nom + torch.sum(den, dim=1) + 1e-8)) / total
    return -torch.log(lp + 1e-8)


def rank_loss(feat_orig, feat_weak, feat_strong):
    """Rank loss: dist(strong, orig) > dist(weak, orig)."""
    d_weak   = F.pairwise_distance(feat_weak, feat_orig, p=2)
    d_strong = F.pairwise_distance(feat_strong, feat_orig, p=2)
    target = torch.ones_like(d_strong)
    return F.binary_cross_entropy_with_logits(d_strong - d_weak, target)


def view_consistency_loss(pred_orig, pred_flip, pred_crop):
    """Variance of predictions across quality-preserving views."""
    preds = torch.stack([pred_orig, pred_flip, pred_crop], dim=1)  # (B, 3)
    return preds.var(dim=1).mean()


def prompt_consistency_loss(pred_orig, pred_para):
    """MSE between prediction with original and paraphrased prompt."""
    return F.mse_loss(pred_orig, pred_para)


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    set_seed(1234)
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    cfg = DATASET_CONFIGS[DATASET]
    csv_path = cfg["splits"][SPLIT]
    
    # Load paraphrased prompts
    para_map = {}
    if os.path.exists(PARAPHRASE_FILE):
        with open(PARAPHRASE_FILE) as f:
            para_map = json.load(f)
        print(f"Loaded {len(para_map)} paraphrased prompts")
    else:
        print("WARNING: No paraphrased prompts found. Prompt consistency loss disabled.")
    
    # Load dataset
    print(f"Loading dataset: {DATASET} / {SPLIT}")
    ds = EnhancedTTADataset(csv_path, cfg)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=enhanced_collate_fn, num_workers=4, pin_memory=True)
    
    # Load Q-Former (with ViT for augmented image encoding)
    print("Loading Q-Former with ViT encoder...")
    ckpt = torch.load(DEFAULT_CHECKPOINT, map_location="cpu", weights_only=False)
    qformer   = QformerWrapper(device, is_eval=False, keep_vit=True).to(device)
    regressor = Regressor(input_dim=768, output_dim=1).to(device)
    proj_head = ProjectionHead(input_dim=768, output_dim=128).to(device)
    
    qformer.model.Qformer.load_state_dict(ckpt["qformer.Qformer"], strict=True)
    qformer.model.query_tokens = nn.Parameter(ckpt["query_tokens"].to(device))
    regressor.load_state_dict(ckpt["regressor"], strict=True)
    
    # Snapshot for reset
    orig_qt = qformer.model.query_tokens.detach().clone()
    ln_mods = [m for m in qformer.model.Qformer.modules() if isinstance(m, nn.LayerNorm)]
    orig_ln = [{"w": m.weight.detach().clone(), "b": m.bias.detach().clone()} for m in ln_mods]
    
    print(f"\nRunning Enhanced TTA: strategy={STRATEGY}, lr={TTA_LR}, steps={TTA_STEPS}")
    print(f"Losses: GC(w={W_GC}) + Rank(w={W_RANK}) + Consistency(w={W_CONSIST}) + Prompt(w={W_PROMPT})")
    print(f"Total batches: {len(loader)}\n")
    
    all_rows = []
    
    for batch in tqdm(loader, desc="Enhanced TTA"):
        image_embeds = batch["image_embeds"].to(device)
        prompts = batch["prompts"]
        descs   = batch["descs"]
        gt_scores = batch["gt_scores"].numpy()
        names = batch["image_names"]
        B = image_embeds.size(0)
        
        # Get paraphrased prompts for this batch
        para_prompts = [para_map.get(p, p) for p in prompts]
        
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
        
        # Extract ViT embeddings for augmented views (frozen, no grad)
        with torch.no_grad():
            embeds_weak   = qformer.extract_image_embeds(batch["img_weak"].to(device))
            embeds_strong = qformer.extract_image_embeds(batch["img_strong"].to(device))
            embeds_flip   = qformer.extract_image_embeds(batch["img_flip"].to(device))
            embeds_crop   = qformer.extract_image_embeds(batch["img_crop"].to(device))
        
        # Setup TTA
        for p in qformer.model.parameters(): p.requires_grad = False
        for p in regressor.parameters(): p.requires_grad = False
        for p in proj_head.parameters(): p.requires_grad = False
        
        ln_params = get_tta_params(qformer, STRATEGY)
        freeze_all_except(qformer, ln_params)
        optimizer = torch.optim.Adam(ln_params, lr=TTA_LR)
        qformer.train()
        
        # Initial preds for GC clustering (detached)
        with torch.no_grad():
            init_mm = qformer.forward_qformer(image_embeds, prompts, descs)
            init_preds = regressor(init_mm).squeeze(-1).detach()
        
        for step in range(1, TTA_STEPS + 1):
            optimizer.zero_grad()
            
            # Forward passes
            mm_orig   = qformer.forward_qformer(image_embeds, prompts, descs)
            mm_weak   = qformer.forward_qformer(embeds_weak, prompts, descs)
            mm_strong = qformer.forward_qformer(embeds_strong, prompts, descs)
            mm_flip   = qformer.forward_qformer(embeds_flip, prompts, descs)
            mm_crop   = qformer.forward_qformer(embeds_crop, prompts, descs)
            
            proj_orig = proj_head(mm_orig)
            proj_weak = proj_head(mm_weak)
            proj_strong = proj_head(mm_strong)
            
            # Predictions for consistency
            pred_orig = regressor(mm_orig).squeeze(-1)
            pred_flip = regressor(mm_flip).squeeze(-1)
            pred_crop = regressor(mm_crop).squeeze(-1)
            
            # 1. GC Loss
            l_gc = gc_loss(proj_orig, init_preds, p=GC_P, temperature=GC_TEMP, device=device)
            
            # 2. Enhanced Rank Loss
            l_rank = rank_loss(proj_orig, proj_weak, proj_strong)
            
            # 3. View Consistency Loss
            l_consist = view_consistency_loss(pred_orig, pred_flip, pred_crop)
            
            # 4. Prompt Consistency Loss
            l_prompt = torch.tensor(0.0, device=device, requires_grad=True)
            if para_map:
                mm_para = qformer.forward_qformer(image_embeds, para_prompts, descs)
                pred_para = regressor(mm_para).squeeze(-1)
                l_prompt = prompt_consistency_loss(pred_orig, pred_para.detach())
            
            total_loss = W_GC * l_gc + W_RANK * l_rank + W_CONSIST * l_consist + W_PROMPT * l_prompt
            
            if total_loss.item() > 0:
                total_loss.backward()
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
        torch.cuda.empty_cache()
    
    # ── Save & Plot ──
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
    
    # Metric progression
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    srcc_vals = [m["srcc"] for m in metrics]
    plcc_vals = [m["plcc"] for m in metrics]
    ax.plot(x, srcc_vals, "o-", color="#2563EB", linewidth=2.5, markersize=10, label="SRCC")
    ax.plot(x, plcc_vals, "s--", color="#DC2626", linewidth=2.5, markersize=10, label="PLCC")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Correlation", fontsize=13)
    ax.set_title("Exp 1: Enhanced TTA (GC+Rank+Consistency+Prompt) on A20K", fontsize=13, fontweight="bold")
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
        ax.set_xlabel("GT Score"); ax.set_ylabel("Predicted Score")
        ax.legend(); ax.grid(True, alpha=0.2)
    plt.suptitle("Exp 1: Enhanced TTA", fontsize=15, fontweight="bold", y=1.01)
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
    ax.set_title(f"Per-Image Error Change (Enhanced TTA)\nImproved: {improved} | Worsened: {worsened}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Error Change (neg=improved)"); ax.set_ylabel("Count")
    ax.legend(); ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "error_histogram.png"), dpi=200)
    plt.close()
    
    print(f"\nAll results saved to {OUT_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
