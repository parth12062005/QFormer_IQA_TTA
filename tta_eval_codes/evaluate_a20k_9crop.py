"""
Evaluate Q-Former on AGIQA-20K using precomputed 9-crop (3x3 grid) embeddings.

Loads precomputed (9, 257, 1408) ViT embeddings per image, runs Q-Former +
Regressor for each crop, and averages the 9 predictions for a robust score.

Supports both Baseline (no TTA) and TTA modes.

Prerequisite:
    python precompute_embeddings_a20k_9crop.py

Usage:
    # Baseline only
    python evaluate_a20k_9crop.py

    # With TTA (GC loss)
    python evaluate_a20k_9crop.py --tta --unfreeze query --tta_lr 1e-4
"""

import os, sys, argparse, random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from lavis.models import load_model_and_preprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tta_framework.param_strategy import get_tta_params, freeze_all_except

# ── Paths ──────────────────────────────────────────────────────────────────
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
_SPLIT_DIR   = os.path.join(
    _PROJECT_DIR,
    "important split files-20260527T062853Z-3-001",
    "important split files",
    "A20K_new",
)

DEFAULT_CHECKPOINT = os.path.join(_PROJECT_DIR, "checkpoints", "evalmi_baseline_qf.pth")
DEFAULT_TEST_CSV   = os.path.join(_SPLIT_DIR, "A20k_test_full_PT1_normalized.csv")
EMBED_ROOT = "/media/parth/021f75bf-bae8-49ef-86a5-28ca19171835/parth/dataset/embeddings/a20k_9crop"

# ── Metrics ────────────────────────────────────────────────────────────────
def _rankdata(a):
    a = np.asarray(a); s = np.argsort(a); inv = np.empty_like(s)
    inv[s] = np.arange(len(a)); a_s = a[s]
    obs = np.concatenate(([True], a_s[1:] != a_s[:-1]))
    dr = np.cumsum(obs); c = np.cumsum(np.bincount(dr))
    return ((c[dr] + c[dr-1] + 1) / 2.0)[inv]

def spearmanr(x, y):
    x, y = np.asarray(x), np.asarray(y)
    rx, ry = _rankdata(x) - _rankdata(x).mean(), _rankdata(y) - _rankdata(y).mean()
    d = np.sqrt(np.sum(rx**2)*np.sum(ry**2))
    return float(np.sum(rx*ry)/d) if d else np.nan

def pearsonr(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    xm, ym = x-x.mean(), y-y.mean()
    d = np.sqrt(np.sum(xm**2)*np.sum(ym**2))
    return float(np.sum(xm*ym)/d) if d else np.nan

def set_seed(s=1234):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ── Regressor ──────────────────────────────────────────────────────────────
class Regressor(nn.Module):
    def __init__(self, input_dim=768, output_dim=1):
        super().__init__()
        self.layer = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.layer(x)

# ── Projection Head ───────────────────────────────────────────────────────
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=768, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, output_dim), nn.Sigmoid())
    def forward(self, x):
        return self.net(x)

# ── Q-Former Wrapper (no ViT needed — embeddings are precomputed) ─────────
class QformerWrapper(nn.Module):
    def __init__(self, device, is_eval=True):
        super().__init__()
        model, _, _ = load_model_and_preprocess(
            name="blip2_feature_extractor", model_type="pretrain",
            is_eval=is_eval, device=device,
        )
        self.model = model.to(device)
        self.device = device
        # Delete ViT to save memory — we use precomputed embeddings
        del self.model.visual_encoder, self.model.ln_vision
        torch.cuda.empty_cache()

    def forward_qformer(self, image_embeds_frozen, prompts, descs):
        B = image_embeds_frozen.size(0)
        image_embeds_frozen = image_embeds_frozen.to(self.device)
        image_atts = torch.ones(image_embeds_frozen.size()[:-1], dtype=torch.long, device=self.device)
        query_tokens = self.model.query_tokens.expand(B, -1, -1)
        text_prompt = self.model.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=self.device)
        mm_mask = torch.cat([query_atts, text_prompt.attention_mask], dim=1)
        mm_out = self.model.Qformer.bert(
            text_prompt.input_ids, query_embeds=query_tokens, attention_mask=mm_mask,
            encoder_hidden_states=image_embeds_frozen, encoder_attention_mask=image_atts, return_dict=True,
        )
        return mm_out.last_hidden_state[:, :query_tokens.size(1), :].mean(dim=1)

# ── Dataset ────────────────────────────────────────────────────────────────
class NineCropEmbedDataset(Dataset):
    """Load precomputed 9-crop embeddings from disk."""
    def __init__(self, csv_path, embed_root):
        self.df = pd.read_csv(csv_path)
        self.df.columns = self.df.columns.str.strip()
        self.embed_root = embed_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row["image_name"])
        embed_name = img_name.replace(".png", ".npz").replace(".jpg", ".npz")
        embed_path = os.path.join(self.embed_root, embed_name)

        # Load (9, 257, 1408) precomputed embeddings
        embeds = torch.from_numpy(np.load(embed_path)["embed"]).float()  # (9, 257, 1408)

        prompt = str(row.get("prompt", ""))
        desc = str(row.get("gen_answer", ""))
        gt = float(row["gt_score"])

        return {
            "embeds_9crop": embeds,
            "prompt": prompt,
            "description": desc,
            "image_name": img_name,
            "gt_score": torch.tensor(gt, dtype=torch.float32),
        }

def collate_9crop(batch):
    return {
        "embeds_9crop": torch.stack([b["embeds_9crop"] for b in batch]),  # (B, 9, 257, 1408)
        "prompts": [b["prompt"] for b in batch],
        "descs": [b["description"] for b in batch],
        "image_names": [b["image_name"] for b in batch],
        "gt_scores": torch.stack([b["gt_score"] for b in batch]),
    }

# ── GC Loss ────────────────────────────────────────────────────────────────
def gc_loss(features, preds, p=0.25, temperature=0.5, device=None):
    B = features.size(0)
    k = max(2, int(B * p))
    if B < 4:
        return torch.tensor(0.0, device=device, requires_grad=True)

    idx = torch.argsort(preds)
    z_i = F.normalize(features[idx[:k]], dim=1)
    z_j = F.normalize(features[idx[-k:]], dim=1)
    n_i, n_j = z_i.size(0), z_j.size(0)

    reps = torch.cat([z_i, z_j], dim=0)
    sim = F.cosine_similarity(reps.unsqueeze(1), reps.unsqueeze(0), dim=2)

    ps_ij = sim[:n_i, :n_j]
    ns_ii = sim[:n_i, :n_i]
    ns_jj = sim[:n_j, :n_j]

    mask_ii = ~torch.eye(n_i, dtype=torch.bool, device=device)
    mask_jj = ~torch.eye(n_j, dtype=torch.bool, device=device)

    logit_ii = torch.exp(ns_ii[mask_ii].view(n_i, n_i - 1) / temperature)
    logit_jj = torch.exp(ns_jj[mask_jj].view(n_j, n_j - 1) / temperature)
    logit_ij = torch.exp(ps_ij / temperature)

    loss_i = -torch.log(logit_ij / (logit_ij + logit_ii.sum(dim=1, keepdim=True) + 1e-8))
    loss_j = -torch.log(logit_ij.t() / (logit_ij.t() + logit_jj.sum(dim=1, keepdim=True) + 1e-8))

    return (loss_i.mean() + loss_j.mean()) / 2

# ── Baseline Evaluation ───────────────────────────────────────────────────
@torch.no_grad()
def run_baseline_9crop(qformer, regressor, dataloader, device):
    """Run baseline (no TTA) with 9-crop averaging using precomputed embeddings."""
    qformer.eval(); regressor.eval()
    all_preds, all_gts, rows = [], [], []

    for batch in tqdm(dataloader, desc="Baseline 9-Crop"):
        embeds = batch["embeds_9crop"].to(device)  # (B, 9, 257, 1408)
        gt = batch["gt_scores"]
        B = embeds.size(0)

        img_preds = []
        for c in range(9):
            crop_embeds = embeds[:, c]  # (B, 257, 1408)
            mm = qformer.forward_qformer(crop_embeds, batch["prompts"], batch["descs"])
            pred = regressor(mm).squeeze(-1)
            img_preds.append(pred)

        img_preds = torch.stack(img_preds, dim=1)  # (B, 9)
        avg_preds = img_preds.mean(dim=1)  # (B,)

        p = avg_preds.float().cpu().numpy()
        g = gt.float().cpu().numpy()
        all_preds.append(p); all_gts.append(g)

        for i in range(B):
            rows.append({
                "image_name": batch["image_names"][i],
                "gt_score": float(g[i]),
                "pred_score": float(p[i]),
            })

    return np.concatenate(all_preds), np.concatenate(all_gts), pd.DataFrame(rows)

# ── TTA Evaluation with 9-Crop ────────────────────────────────────────────
def run_tta_9crop(qformer, regressor, proj_head, dataloader, device,
                  unfreeze_strategy, tta_steps, tta_lr):
    """Run TTA with 9-crop averaging using precomputed embeddings."""
    all_preds_base, all_preds_tta, all_gts, rows = [], [], [], []

    # Snapshot original state for reset
    orig_query_tokens = qformer.model.query_tokens.detach().clone()
    qf_layernorms = [m for m in qformer.model.Qformer.modules() if isinstance(m, nn.LayerNorm)]
    orig_ln_states = [
        {"weight": m.weight.detach().clone() if m.weight is not None else None,
         "bias": m.bias.detach().clone() if m.bias is not None else None}
        for m in qf_layernorms
    ]

    for batch in tqdm(dataloader, desc="TTA 9-Crop"):
        embeds = batch["embeds_9crop"].to(device)  # (B, 9, 257, 1408)
        gt = batch["gt_scores"]
        B = embeds.size(0)

        # 1. Reset model state
        with torch.no_grad():
            qformer.model.query_tokens.copy_(orig_query_tokens)
            for m, state in zip(qf_layernorms, orig_ln_states):
                if m.weight is not None and state["weight"] is not None:
                    m.weight.copy_(state["weight"])
                if m.bias is not None and state["bias"] is not None:
                    m.bias.copy_(state["bias"])

        # 2. Baseline predictions (before TTA)
        qformer.eval()
        with torch.no_grad():
            base_crop_preds = []
            for c in range(9):
                mm = qformer.forward_qformer(embeds[:, c], batch["prompts"], batch["descs"])
                pred = regressor(mm).squeeze(-1)
                base_crop_preds.append(pred)
            base_preds = torch.stack(base_crop_preds, dim=1).mean(dim=1)

        # 3. Get initial predictions for GC loss clustering
        with torch.no_grad():
            init_mm_list = []
            for c in range(9):
                mm = qformer.forward_qformer(embeds[:, c], batch["prompts"], batch["descs"])
                init_mm_list.append(mm)
            init_mm_avg = torch.stack(init_mm_list, dim=1).mean(dim=1)
            init_preds = regressor(init_mm_avg).squeeze(-1).detach()

        # 4. TTA optimization
        params_to_update = get_tta_params(qformer, unfreeze_strategy)
        freeze_all_except(qformer, params_to_update)
        if len(params_to_update) == 0:
            p_base = base_preds.float().cpu().numpy()
            g = gt.float().cpu().numpy()
            all_preds_base.append(p_base)
            all_preds_tta.append(p_base)
            all_gts.append(g)
            continue

        optimizer = optim.Adam(params_to_update, lr=tta_lr)

        qformer.train()
        for step in range(tta_steps):
            optimizer.zero_grad()

            mm_list = []
            for c in range(9):
                mm = qformer.forward_qformer(embeds[:, c], batch["prompts"], batch["descs"])
                mm_list.append(mm)
            mm_avg = torch.stack(mm_list, dim=1).mean(dim=1)

            proj_feats = proj_head(mm_avg)
            loss = gc_loss(proj_feats, init_preds, device=device)

            if loss.item() > 0:
                loss.backward()
                optimizer.step()

        # 5. Final predictions after TTA
        qformer.eval()
        with torch.no_grad():
            tta_crop_preds = []
            for c in range(9):
                mm = qformer.forward_qformer(embeds[:, c], batch["prompts"], batch["descs"])
                pred = regressor(mm).squeeze(-1)
                tta_crop_preds.append(pred)
            tta_preds = torch.stack(tta_crop_preds, dim=1).mean(dim=1)

        p_base = base_preds.float().cpu().numpy()
        p_tta = tta_preds.float().cpu().numpy()
        g = gt.float().cpu().numpy()
        all_preds_base.append(p_base)
        all_preds_tta.append(p_tta)
        all_gts.append(g)

        for i in range(B):
            rows.append({
                "image_name": batch["image_names"][i],
                "gt_score": float(g[i]),
                "pred_baseline": float(p_base[i]),
                "pred_tta": float(p_tta[i]),
            })

    preds_base = np.concatenate(all_preds_base)
    preds_tta = np.concatenate(all_preds_tta)
    gts = np.concatenate(all_gts)
    return preds_base, preds_tta, gts, pd.DataFrame(rows)

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="9-Crop Q-Former Evaluation on A20K (precomputed)")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--csv", type=str, default=DEFAULT_TEST_CSV)
    parser.add_argument("--embed_root", type=str, default=EMBED_ROOT)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default=os.path.join(_SCRIPT_DIR, "results", "a20k_9crop"))

    # TTA options
    parser.add_argument("--tta", action="store_true", help="Enable TTA")
    parser.add_argument("--unfreeze", type=str, default="query",
                        choices=["none", "layernorm", "self_attn_ln", "query", "both",
                                 "crossattn_query", "crossattn_query_026"])
    parser.add_argument("--tta_steps", type=int, default=3)
    parser.add_argument("--tta_lr", type=float, default=1e-4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  9-Crop Q-Former Evaluation on A20K (Precomputed Embeddings)")
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  CSV:         {args.csv}")
    print(f"  Embed Root:  {args.embed_root}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  TTA:         {args.tta}")
    if args.tta:
        print(f"  Unfreeze:    {args.unfreeze}")
        print(f"  TTA steps:   {args.tta_steps}")
        print(f"  TTA LR:      {args.tta_lr}")
    print(f"  Output:      {args.output_dir}")
    print("=" * 60)

    # Load model (no ViT needed)
    print("Loading Q-Former model (no ViT — using precomputed embeddings)...")
    qformer = QformerWrapper(device=device, is_eval=True).to(device)
    regressor = Regressor(input_dim=768, output_dim=1).to(device)
    proj_head = ProjectionHead(input_dim=768, output_dim=128).to(device)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    qformer.model.Qformer.load_state_dict(ckpt["qformer.Qformer"], strict=True)
    qformer.model.query_tokens = nn.Parameter(ckpt["query_tokens"].to(device))
    regressor.load_state_dict(ckpt["regressor"], strict=True)
    if "proj_head" in ckpt:
        proj_head.load_state_dict(ckpt["proj_head"])
    print("Checkpoint loaded.")

    # Dataset
    ds = NineCropEmbedDataset(args.csv, args.embed_root)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_9crop, num_workers=args.num_workers, pin_memory=True)
    print(f"Dataset: {len(ds)} images, 9 crops each")

    if not args.tta:
        preds, gts, df = run_baseline_9crop(qformer, regressor, loader, device)
        srcc = spearmanr(preds, gts)
        plcc = pearsonr(preds, gts)

        print("\n" + "=" * 60)
        print(f"  [Baseline 9-Crop]  SRCC: {srcc:.6f}  |  PLCC: {plcc:.6f}")
        print("=" * 60)

        df.to_csv(os.path.join(args.output_dir, "baseline_9crop.csv"), index=False)
    else:
        preds_base, preds_tta, gts, df = run_tta_9crop(
            qformer, regressor, proj_head, loader, device,
            args.unfreeze, args.tta_steps, args.tta_lr,
        )
        srcc_base = spearmanr(preds_base, gts)
        plcc_base = pearsonr(preds_base, gts)
        srcc_tta = spearmanr(preds_tta, gts)
        plcc_tta = pearsonr(preds_tta, gts)

        print("\n" + "=" * 60)
        print(f"  [Baseline 9-Crop]  SRCC: {srcc_base:.6f}  |  PLCC: {plcc_base:.6f}")
        print(f"  [TTA 9-Crop]       SRCC: {srcc_tta:.6f}  |  PLCC: {plcc_tta:.6f}")
        print(f"  [Delta]            SRCC: {srcc_tta - srcc_base:+.6f}  |  PLCC: {plcc_tta - plcc_base:+.6f}")
        print("=" * 60)

        df.to_csv(os.path.join(args.output_dir, "tta_9crop.csv"), index=False)

    print(f"\n[Saved] Results in: {args.output_dir}")

if __name__ == "__main__":
    set_seed(1234)
    main()
