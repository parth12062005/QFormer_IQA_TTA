"""
Evaluate baseline Q-Former checkpoint on AGIQA-20K dataset (all splits).

Reports per-split and combined SRCC + PLCC.
Saves:
  - Per-image predictions CSV for each split
  - Combined per-image CSV (all splits)
  - Summary CSV with SRCC/PLCC per split and combined

Usage:
    python evaluate_a20k_baseline.py [--checkpoint PATH] [--batch_size N]

NOTE: Uses PRECOMPUTED ViT embeddings. Run precompute_embeddings_a20k.py first.
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from lavis.models import load_model_and_preprocess


##### ------------- ####
#####  DEFAULTS
##### ------------- ####
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)  # parent: QFormer_IQA_TTA
DEFAULT_CHECKPOINT = os.path.join(_PROJECT_DIR, "checkpoints", "evalmi_baseline_qf.pth")

# Split CSVs from the important split files folder
_SPLIT_DIR = os.path.join(
    _PROJECT_DIR,
    "important split files-20260527T062853Z-3-001",
    "important split files",
    "A20K_new",
)
DEFAULT_TRAIN_CSV = os.path.join(_SPLIT_DIR, "A20k_train_full_PT1_normalized.csv")
DEFAULT_VAL_CSV   = os.path.join(_SPLIT_DIR, "A20k_val_full_PT1_normalized.csv")
DEFAULT_TEST_CSV  = os.path.join(_SPLIT_DIR, "A20k_test_full_PT1_normalized.csv")

# A20K embeddings are flat files (e.g. DALLE2_0000.npz) saved directly in a20k/
EMBED_ROOT = "/media/parth/021f75bf-bae8-49ef-86a5-28ca19171835/parth/dataset/embeddings/a20k"

IMG_COL    = "image_name"
PROMPT_COL = "prompt"
DESC_COL   = "gen_answer"
GT_COL     = "gt_score"

DEFAULT_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "results", "a20k_baseline_splits")


##### ------------- ####
#####  UTILS
##### ------------- ####
def set_seed(seed: int):
    import random
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
    def __init__(self, csv_path, embed_root=EMBED_ROOT):
        self.df = pd.read_csv(csv_path)
        self.embed_root = embed_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row[IMG_COL])

        # A20K images have flat names like DALLE2_0000.png -> DALLE2_0000.npz
        embed_path = os.path.join(
            self.embed_root,
            img_name.replace(".png", ".npz").replace(".jpg", ".npz"),
        )
        image_embeds = torch.from_numpy(np.load(embed_path)["embed"]).float()

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
class Regressor(nn.Module):
    """Single-layer linear regressor (matches checkpoint architecture)."""
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.layer(x)


class QformerWrapper(nn.Module):
    """
    Q-Former wrapper that takes PRECOMPUTED image embeddings instead of raw images.
    No ViT encoder needed — only loads the Q-Former + query tokens.
    """
    def __init__(self, device, is_eval=True):
        super().__init__()
        model, _, _ = load_model_and_preprocess(
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
#####  EVALUATION LOOP
##### ------------- ####
@torch.no_grad()
def evaluate(qformer, regressor, dataloader, device, desc_tag="Evaluating"):
    """Run inference and collect predictions + ground truths."""
    qformer.eval()
    regressor.eval()

    all_preds, all_gts = [], []
    rows = []

    for batch in tqdm(dataloader, desc=desc_tag):
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
        description="Evaluate baseline Q-Former checkpoint on AGIQA-20K (all splits)"
    )
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help="Path to the .pth checkpoint file")
    parser.add_argument("--train_csv", type=str, default=DEFAULT_TRAIN_CSV,
                        help="Path to the train split CSV")
    parser.add_argument("--val_csv", type=str, default=DEFAULT_VAL_CSV,
                        help="Path to the val split CSV")
    parser.add_argument("--test_csv", type=str, default=DEFAULT_TEST_CSV,
                        help="Path to the test split CSV")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Device ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load models ---
    print("Loading Q-Former model...")
    qformer   = QformerWrapper(device=device, is_eval=True).to(device)
    regressor = Regressor(input_dim=768, output_dim=1).to(device)

    # --- Load checkpoint ---
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    qformer.model.Qformer.load_state_dict(ckpt["qformer.Qformer"], strict=True)
    qformer.model.query_tokens = nn.Parameter(ckpt["query_tokens"].to(device))
    regressor.load_state_dict(ckpt["regressor"], strict=True)
    print("Checkpoint loaded successfully.")

    qformer.eval()
    regressor.eval()

    # --- Evaluate each split ---
    splits = {
        "train": args.train_csv,
        "val":   args.val_csv,
        "test":  args.test_csv,
    }

    split_results = {}           # split_name -> (srcc, plcc, n_samples)
    combined_preds_list = []     # for combined metrics
    combined_gts_list   = []
    all_split_dfs       = []     # for combined per-image CSV

    for split_name, csv_path in splits.items():
        print(f"\n{'='*60}")
        print(f"  Evaluating split: {split_name.upper()}")
        print(f"  CSV: {csv_path}")
        print(f"{'='*60}")

        dataset = QFormerEmbeddingDataset(csv_path=csv_path)
        loader  = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        print(f"  Samples: {len(dataset)}")

        all_preds, all_gts, df = evaluate(
            qformer, regressor, loader, device,
            desc_tag=f"Eval {split_name}",
        )

        srcc = spearmanr_numpy(all_preds, all_gts)
        plcc = pearsonr_numpy(all_preds, all_gts)

        split_results[split_name] = (srcc, plcc, len(dataset))

        # Add split column for per-image tracking
        df.insert(0, "split", split_name)

        # Save per-split per-image predictions
        per_sample_csv = os.path.join(args.output_dir, f"a20k_{split_name}_predictions.csv")
        df.to_csv(per_sample_csv, index=False)
        print(f"  [Saved] {per_sample_csv}")
        print(f"  SRCC: {srcc:.6f}  |  PLCC: {plcc:.6f}")

        # Accumulate for combined
        combined_preds_list.append(all_preds)
        combined_gts_list.append(all_gts)
        all_split_dfs.append(df)

    # --- Combined metrics ---
    combined_preds = np.concatenate(combined_preds_list)
    combined_gts   = np.concatenate(combined_gts_list)
    combined_srcc  = spearmanr_numpy(combined_preds, combined_gts)
    combined_plcc  = pearsonr_numpy(combined_preds, combined_gts)
    combined_n     = len(combined_preds)

    # --- Print summary table ---
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS — AGIQA-20K Baseline Q-Former")
    print("=" * 60)
    print(f"  {'Split':<10} {'N':>8} {'SRCC':>10} {'PLCC':>10}")
    print("-" * 60)
    for split_name, (srcc, plcc, n) in split_results.items():
        print(f"  {split_name:<10} {n:>8} {srcc:>10.6f} {plcc:>10.6f}")
    print("-" * 60)
    print(f"  {'Combined':<10} {combined_n:>8} {combined_srcc:>10.6f} {combined_plcc:>10.6f}")
    print("=" * 60)

    # --- Save combined per-image CSV (all splits) ---
    combined_df = pd.concat(all_split_dfs, ignore_index=True)
    combined_images_csv = os.path.join(args.output_dir, "a20k_baseline_all_images.csv")
    combined_df.to_csv(combined_images_csv, index=False)
    print(f"\n[Saved] Combined per-image CSV ({combined_n} images): {combined_images_csv}")

    # --- Save summary CSV ---
    summary_rows = []
    for split_name, (srcc, plcc, n) in split_results.items():
        summary_rows.append({
            "split": split_name,
            "n_samples": n,
            "srcc": srcc,
            "plcc": plcc,
        })
    summary_rows.append({
        "split": "combined",
        "n_samples": combined_n,
        "srcc": combined_srcc,
        "plcc": combined_plcc,
    })

    summary_csv = os.path.join(args.output_dir, "a20k_baseline_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"[Saved] Summary CSV: {summary_csv}")


if __name__ == "__main__":
    set_seed(1234)
    main()
