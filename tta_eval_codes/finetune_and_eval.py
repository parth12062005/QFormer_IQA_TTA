"""
Fine-tune EvalMI-pretrained Q-Former on a target dataset and evaluate.

Protocol (from the paper table):
  1. Load Q-Former checkpoint pre-trained on EvalMI
  2. Sample X% of the target dataset's TRAINING split (5%, 10%, or 20%)
  3. Fine-tune Q-Former + query tokens + regressor on this subset
  4. Evaluate on the FULL test split
  5. Report SRCC and PLCC

Supports:
  - AIGIQA-20K (a20k)   — SRCC metric, .npz embeddings
  - AGIQA-3K  (a3k)     — SRCC metric, .npy embeddings

Auto-detects regressor architecture (1-layer linear vs 2-layer MLP) from checkpoint.

Usage:
    python finetune_and_eval.py --dataset a20k --fraction 0.1 --checkpoint /path/to/ckpt.pth
    python finetune_and_eval.py --dataset a3k  --fraction 0.05 --checkpoint /path/to/ckpt.pth

NOTE: Uses PRECOMPUTED ViT embeddings. Run precompute_embeddings_{a20k,a3k}.py first.
"""

import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from lavis.models import load_model_and_preprocess


##### ------------- ####
#####  PATHS
##### ------------- ####
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)

_SPLIT_DIR = os.path.join(
    _PROJECT_DIR,
    "important split files-20260527T062853Z-3-001",
    "important split files",
)

# ── Dataset configs ──
DATASET_CONFIGS = {
    "a20k": {
        "train_csv": os.path.join(_SPLIT_DIR, "A20K_new", "A20k_train_full_PT1_normalized.csv"),
        "val_csv":   os.path.join(_SPLIT_DIR, "A20K_new", "A20k_val_full_PT1_normalized.csv"),
        "test_csv":  os.path.join(_SPLIT_DIR, "A20K_new", "A20k_test_full_PT1_normalized.csv"),
        "embed_root": "/media/parth/021f75bf-bae8-49ef-86a5-28ca19171835/parth/dataset/embeddings/a20k",
        "embed_format": "npz",   # .npz with key 'embed'
        "name": "AIGIQA-20K",
    },
    "a3k": {
        "train_csv": os.path.join(_SPLIT_DIR, "A3K_new", "a3k_train_full_gen_responses_PT1_normalized.csv"),
        "val_csv":   os.path.join(_SPLIT_DIR, "A3K_new", "a3k_val_full_gen_responses_PT1_normalized.csv"),
        "test_csv":  os.path.join(_SPLIT_DIR, "A3K_new", "a3k_test_full_gen_responses_PT1_normalized.csv"),
        "embed_root": "/media/parth/021f75bf-bae8-49ef-86a5-28ca19171835/parth/dataset/embeddings/a3k",
        "embed_format": "npy",   # .npy (raw numpy array)
        "name": "AGIQA-3K",
    },
}

DEFAULT_CHECKPOINT = os.path.join(_PROJECT_DIR, "checkpoints", "evalmi_baseline_qf.pth")

IMG_COL    = "image_name"
PROMPT_COL = "prompt"
DESC_COL   = "gen_answer"
GT_COL     = "gt_score"


##### ------------- ####
#####  UTILS
##### ------------- ####
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


##### ------------- ####
#####  METRICS
##### ------------- ####
def rankdata_numpy(a):
    """NumPy-only equivalent of scipy.stats.rankdata(method='average')."""
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
    """Spearman Rank Correlation Coefficient (SRCC)."""
    x, y = np.asarray(x), np.asarray(y)
    assert x.shape == y.shape, "x and y must have same shape"
    rx, ry = rankdata_numpy(x), rankdata_numpy(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = np.sqrt(np.sum(rx**2) * np.sum(ry**2))
    return np.nan if denom == 0 else float(np.sum(rx * ry) / denom)


def pearsonr_numpy(x, y):
    """Pearson Linear Correlation Coefficient (PLCC)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    assert x.shape == y.shape, "x and y must have same shape"
    xm, ym = x - x.mean(), y - y.mean()
    denom = np.sqrt(np.sum(xm**2) * np.sum(ym**2))
    return np.nan if denom == 0 else float(np.sum(xm * ym) / denom)


##### ------------- ####
#####  DATASET + COLLATE
##### ------------- ####
class QFormerEmbeddingDataset(Dataset):
    """Loads precomputed ViT embeddings instead of raw images."""
    def __init__(self, csv_path=None, df=None, embed_root="", embed_format="npz"):
        assert (csv_path is not None) ^ (df is not None), "Provide exactly one of csv_path or df"
        self.df = pd.read_csv(csv_path) if csv_path is not None else df.reset_index(drop=True)
        self.df.columns = self.df.columns.str.strip()
        self.embed_root = embed_root
        self.embed_format = embed_format

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row[IMG_COL])

        if self.embed_format == "npz":
            embed_path = os.path.join(
                self.embed_root,
                img_name.replace(".png", ".npz").replace(".jpg", ".npz"),
            )
            image_embeds = torch.from_numpy(np.load(embed_path)["embed"]).float()
        else:  # npy
            embed_path = os.path.join(
                self.embed_root,
                img_name.replace(".png", ".npy").replace(".jpg", ".npy"),
            )
            image_embeds = torch.from_numpy(np.load(embed_path)).float()

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
    return {
        "image_embeds": torch.stack([b["image_embeds"] for b in batch], dim=0),
        "prompts":      [b["prompt"] for b in batch],
        "descs":        [b["description"] for b in batch],
        "image_names":  [b["image_name"] for b in batch],
        "gt_scores":    torch.stack([b["gt_score"] for b in batch], dim=0),
    }


##### ------------- ####
#####  MODELS
##### ------------- ####
class RegressorLinear(nn.Module):
    """Single-layer linear regressor: 768 -> 1"""
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.layer(x)


class RegressorMLP(nn.Module):
    """Two-layer MLP regressor: 768 -> hidden_dim -> 1"""
    def __init__(self, input_dim, hidden_dim=256, output_dim=1):
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
    Q-Former wrapper that takes PRECOMPUTED image embeddings.
    No ViT encoder needed.
    """
    def __init__(self, device, is_eval=False):
        super().__init__()
        model, _, _ = load_model_and_preprocess(
            name="blip2_feature_extractor",
            model_type="pretrain",
            is_eval=is_eval,
            device=device,
        )
        self.model = model.to(device)
        self.device = device

        # Free ViT from GPU memory
        del self.model.visual_encoder
        del self.model.ln_vision
        torch.cuda.empty_cache()

    def forward(self, image_embeds_frozen, prompts, descs):
        B = image_embeds_frozen.size(0)
        image_embeds_frozen = image_embeds_frozen.to(self.device)

        image_atts = torch.ones(
            image_embeds_frozen.size()[:-1], dtype=torch.long, device=self.device
        )

        query_tokens = self.model.query_tokens.expand(B, -1, -1)

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

        mm_query_embeds = mm_out.last_hidden_state[:, : query_tokens.size(1), :]
        mm_mean_embeds = mm_query_embeds.mean(dim=1)

        return mm_mean_embeds


##### ------------- ####
#####  CHECKPOINT UTILS
##### ------------- ####
def detect_regressor_type(ckpt):
    """Auto-detect regressor architecture from checkpoint keys."""
    reg_keys = list(ckpt["regressor"].keys())
    if "layer.weight" in reg_keys:
        return "linear"
    elif "layer.0.weight" in reg_keys:
        return "mlp"
    else:
        raise ValueError(f"Unknown regressor architecture. Keys: {reg_keys}")


def create_regressor(reg_type, ckpt):
    """Create regressor matching the checkpoint architecture."""
    if reg_type == "linear":
        return RegressorLinear(input_dim=768, output_dim=1)
    else:
        hidden_dim = ckpt["regressor"]["layer.0.weight"].shape[0]
        return RegressorMLP(input_dim=768, hidden_dim=hidden_dim, output_dim=1)


##### ------------- ####
#####  TRAIN / EVAL LOOPS
##### ------------- ####
def train_one_epoch(qformer, regressor, dataloader, optimizer, criterion, device):
    qformer.train()
    regressor.train()

    total_loss = 0.0
    pbar = tqdm(dataloader, desc="  Training")
    for step, batch in enumerate(pbar, 1):
        image_embeds = batch["image_embeds"].to(device)
        prompts = batch["prompts"]
        descs = batch["descs"]
        gt_scores = batch["gt_scores"].to(device)

        optimizer.zero_grad()

        mm_mean_embeds = qformer(image_embeds, prompts, descs)
        pred = regressor(mm_mean_embeds).squeeze(-1)
        loss = criterion(pred, gt_scores)

        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().item())
        pbar.set_postfix(loss=f"{total_loss / step:.6f}")

    return total_loss / max(1, len(dataloader))


@torch.no_grad()
def evaluate(qformer, regressor, dataloader, device, desc_tag="Evaluating"):
    """Run inference and return predictions, ground truths, and per-image DataFrame."""
    qformer.eval()
    regressor.eval()

    all_preds, all_gts = [], []
    rows = []

    for batch in tqdm(dataloader, desc=f"  {desc_tag}"):
        image_embeds = batch["image_embeds"].to(device, non_blocking=True)
        prompts      = batch["prompts"]
        descs        = batch["descs"]
        gt_scores    = batch["gt_scores"].to(device, non_blocking=True)

        mm_mean_embeds = qformer(image_embeds, prompts, descs)
        pred = regressor(mm_mean_embeds).squeeze(-1)

        pred_cpu = pred.float().cpu().numpy()
        gt_cpu   = gt_scores.float().cpu().numpy()

        all_preds.append(pred_cpu)
        all_gts.append(gt_cpu)

        for i in range(len(batch["image_names"])):
            rows.append({
                "image_name": batch["image_names"][i],
                "prompt":     prompts[i],
                "gen_answer": descs[i],
                "gt_score":   float(gt_cpu[i]),
                "pred_score": float(pred_cpu[i]),
            })

    all_preds = np.concatenate(all_preds)
    all_gts   = np.concatenate(all_gts)

    return all_preds, all_gts, pd.DataFrame(rows)


##### ------------- ####
#####  MAIN
##### ------------- ####
def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune EvalMI-pretrained Q-Former on target dataset with X% labels"
    )
    parser.add_argument("--dataset", type=str, required=True, choices=["a20k", "a3k"],
                        help="Target dataset: a20k or a3k")
    parser.add_argument("--fraction", type=float, required=True,
                        help="Fraction of training data to use (0.0 for zero-shot, 0.05, 0.10, 0.20)")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help="Path to EvalMI-pretrained checkpoint (.pth)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (auto-generated if not specified)")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=256,
                        help="Evaluation batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    set_seed(args.seed)

    # ── Resolve dataset config ──
    cfg = DATASET_CONFIGS[args.dataset]
    frac_pct = int(args.fraction * 100)

    # ── Auto-generate output dir ──
    if args.output_dir is None:
        ckpt_tag = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.output_dir = os.path.join(
            _SCRIPT_DIR, "results",
            f"finetune_{args.dataset}_{ckpt_tag}_{frac_pct}pct"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Device ──
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print(f"  FINE-TUNE & EVALUATE: {cfg['name']}")
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  Fraction:    {args.fraction} ({frac_pct}%)")
    print(f"  Epochs:      {args.epochs}")
    print(f"  LR:          {args.lr}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Seed:        {args.seed}")
    print(f"  Device:      {device}")
    print(f"  Output dir:  {args.output_dir}")
    print("=" * 70)

    # ── Load checkpoint and detect architecture ──
    print("\nLoading checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    reg_type = detect_regressor_type(ckpt)
    print(f"  Regressor type: {reg_type}")

    # ── Create models ──
    print("Loading Q-Former model...")
    qformer   = QformerWrapper(device=device, is_eval=False).to(device)
    regressor = create_regressor(reg_type, ckpt).to(device)

    # ── Load checkpoint weights ──
    qformer.model.Qformer.load_state_dict(ckpt["qformer.Qformer"], strict=True)
    qformer.model.query_tokens = nn.Parameter(ckpt["query_tokens"].to(device))
    regressor.load_state_dict(ckpt["regressor"], strict=True)
    print("  Checkpoint loaded successfully.")

    # ── Freeze/unfreeze ──
    for p in qformer.model.parameters():
        p.requires_grad = False
    qformer.model.query_tokens.requires_grad = True
    for p in qformer.model.Qformer.parameters():
        p.requires_grad = True
    for p in regressor.parameters():
        p.requires_grad = True

    # ── Prepare data ──
    print(f"\nPreparing data ({cfg['name']})...")

    # Sample fraction of training data
    train_df_full = pd.read_csv(cfg["train_csv"])
    train_df_full.columns = train_df_full.columns.str.strip()
    train_df_sampled = train_df_full.sample(frac=args.fraction, random_state=args.seed).reset_index(drop=True)

    if args.fraction > 0.0:
        train_dataset = QFormerEmbeddingDataset(
            df=train_df_sampled, embed_root=cfg["embed_root"], embed_format=cfg["embed_format"]
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
        )
    else:
        train_dataset = []
        train_loader = []

    val_dataset = QFormerEmbeddingDataset(
        csv_path=cfg["val_csv"], embed_root=cfg["embed_root"], embed_format=cfg["embed_format"]
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.eval_batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
    )

    test_dataset = QFormerEmbeddingDataset(
        csv_path=cfg["test_csv"], embed_root=cfg["embed_root"], embed_format=cfg["embed_format"]
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.eval_batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
    )

    print(f"  Train samples (full):    {len(train_df_full)}")
    print(f"  Train samples ({frac_pct}%):   {len(train_df_sampled)}")
    print(f"  Val samples:             {len(val_dataset)}")
    print(f"  Test samples:            {len(test_dataset)}")

    # ── Quick zero-shot handling ──
    if args.fraction == 0.0:
        print("\nFraction is 0.0 — Running zero-shot evaluation without training...")
        test_pred_csv = os.path.join(args.output_dir, f"{args.dataset}_test_predictions_zeroshot.csv")
        test_preds, test_gts, test_df = evaluate(
            qformer, regressor, test_loader, device, desc_tag="Test"
        )
        test_srcc = spearmanr_numpy(test_preds, test_gts)
        test_plcc = pearsonr_numpy(test_preds, test_gts)
        
        test_df.to_csv(test_pred_csv, index=False)
        print("\n" + "=" * 70)
        print(f"  FINAL RESULTS (ZERO-SHOT) — {cfg['name']}")
        print("=" * 70)
        print(f"  Checkpoint:   {os.path.basename(args.checkpoint)}")
        print(f"  Regressor:    {reg_type}")
        print(f"  Test SRCC:    {test_srcc:.6f}")
        print(f"  Test PLCC:    {test_plcc:.6f}")
        print("=" * 70)
        
        summary_csv = os.path.join(args.output_dir, "summary.csv")
        pd.DataFrame([{
            "dataset": args.dataset,
            "dataset_name": cfg["name"],
            "checkpoint": os.path.basename(args.checkpoint),
            "regressor_type": reg_type,
            "fraction": args.fraction,
            "fraction_pct": 0,
            "train_total": len(train_df_full),
            "train_used": 0,
            "test_samples": len(test_dataset),
            "best_epoch": 0,
            "best_val_srcc": -1.0,
            "test_srcc": test_srcc,
            "test_plcc": test_plcc,
            "lr": args.lr,
            "epochs": 0,
            "batch_size": args.batch_size,
            "seed": args.seed,
        }]).to_csv(summary_csv, index=False)
        
        print(f"\n[RESULT] dataset={args.dataset} fraction=0% ckpt={os.path.basename(args.checkpoint)} SRCC={test_srcc:.6f} PLCC={test_plcc:.6f}")
        return

    # ── Optimizer + Loss ──
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        [p for p in qformer.model.parameters() if p.requires_grad] + list(regressor.parameters()),
        lr=args.lr,
    )

    # ── Training loop ──
    best_val_srcc  = -1.0
    best_test_srcc = -1.0
    best_test_plcc = -1.0
    best_epoch     = -1

    test_pred_csv = os.path.join(args.output_dir, f"{args.dataset}_test_predictions.csv")

    for epoch in range(1, args.epochs + 1):
        print(f"\n── Epoch {epoch}/{args.epochs} ──")

        train_loss = train_one_epoch(qformer, regressor, train_loader, optimizer, criterion, device)
        print(f"  Train loss: {train_loss:.6f}")

        # Validate
        val_preds, val_gts, _ = evaluate(qformer, regressor, val_loader, device, desc_tag="Val")
        val_srcc = spearmanr_numpy(val_preds, val_gts)
        val_plcc = pearsonr_numpy(val_preds, val_gts)
        print(f"  Val  SRCC: {val_srcc:.6f}  |  PLCC: {val_plcc:.6f}")

        if val_srcc > best_val_srcc:
            best_val_srcc = val_srcc

            # Test
            test_preds, test_gts, test_df = evaluate(
                qformer, regressor, test_loader, device, desc_tag="Test"
            )
            test_srcc = spearmanr_numpy(test_preds, test_gts)
            test_plcc = pearsonr_numpy(test_preds, test_gts)

            best_test_srcc = test_srcc
            best_test_plcc = test_plcc
            best_epoch = epoch

            # Save test predictions
            test_df.to_csv(test_pred_csv, index=False)

            print(f"  ★ New best! Test SRCC: {test_srcc:.6f}  |  PLCC: {test_plcc:.6f}")

        print(f"  Best val SRCC:  {best_val_srcc:.6f}")
        print(f"  Best test SRCC: {best_test_srcc:.6f}  |  PLCC: {best_test_plcc:.6f}  (epoch {best_epoch})")

    # ── Final summary ──
    print("\n" + "=" * 70)
    print(f"  FINAL RESULTS — {cfg['name']}  |  {frac_pct}% labels")
    print("=" * 70)
    ckpt_basename = os.path.basename(args.checkpoint)
    print(f"  Checkpoint:   {ckpt_basename}")
    print(f"  Regressor:    {reg_type}")
    print(f"  Train subset: {len(train_df_sampled)} / {len(train_df_full)} ({frac_pct}%)")
    print(f"  Best epoch:   {best_epoch}")
    print(f"  Test SRCC:    {best_test_srcc:.6f}")
    print(f"  Test PLCC:    {best_test_plcc:.6f}")
    print(f"  Predictions:  {test_pred_csv}")
    print("=" * 70)

    # ── Save summary CSV ──
    summary_csv = os.path.join(args.output_dir, "summary.csv")
    pd.DataFrame([{
        "dataset": args.dataset,
        "dataset_name": cfg["name"],
        "checkpoint": ckpt_basename,
        "regressor_type": reg_type,
        "fraction": args.fraction,
        "fraction_pct": frac_pct,
        "train_total": len(train_df_full),
        "train_used": len(train_df_sampled),
        "test_samples": len(test_dataset),
        "best_epoch": best_epoch,
        "best_val_srcc": best_val_srcc,
        "test_srcc": best_test_srcc,
        "test_plcc": best_test_plcc,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }]).to_csv(summary_csv, index=False)
    print(f"[Saved] Summary: {summary_csv}")

    # Print machine-parseable result line for shell script
    print(f"\n[RESULT] dataset={args.dataset} fraction={frac_pct}% ckpt={ckpt_basename} SRCC={best_test_srcc:.6f} PLCC={best_test_plcc:.6f}")


if __name__ == "__main__":
    main()
