"""
Evaluate a baseline Q-Former checkpoint on the EvalMi-50K test set.

Computes:
  - SRCC  (Spearman Rank Correlation Coefficient)
  - PLCC  (Pearson Linear Correlation Coefficient)
  - Accuracy (percentage of predictions within a threshold of ground truth)

Usage:
    python evaluate_baseline.py [--checkpoint PATH] [--test_csv PATH] [--batch_size N]

NOTE: Uses PRECOMPUTED ViT embeddings. Run precompute_embeddings.py first.
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
DEFAULT_CHECKPOINT = os.path.join(_SCRIPT_DIR, "checkpoints", "evalmi_baseline_qf.pth")
DEFAULT_TRAIN_CSV  = os.path.normpath(os.path.join(_SCRIPT_DIR, "../EvalMi-50K/evalmi_train.csv"))
DEFAULT_VAL_CSV    = os.path.normpath(os.path.join(_SCRIPT_DIR, "../EvalMi-50K/evalmi_val.csv"))
DEFAULT_TEST_CSV   = os.path.normpath(os.path.join(_SCRIPT_DIR, "../EvalMi-50K/evalmi_test.csv"))
EMBED_ROOT         = "/media/parth/Balance/parth/dataset/embeddings"

IMG_COL    = "image_name"
PROMPT_COL = "prompt"
DESC_COL   = "gen_answer"
GT_COL     = "gt_score"


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


def accuracy_at_threshold(pred, gt, thresholds=(5.0, 10.0, 15.0)):
    """
    Compute accuracy as the percentage of predictions within each threshold
    of the ground truth score.

    Returns:
        dict mapping threshold -> accuracy percentage
    """
    pred, gt = np.asarray(pred), np.asarray(gt)
    abs_diff = np.abs(pred - gt)
    return {thr: float(np.mean(abs_diff <= thr) * 100.0) for thr in thresholds}


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
#####  MODEL
##### ------------- ####
class Regressor(nn.Module):
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
def evaluate(qformer, regressor, dataloader, device):
    """Run inference and collect predictions + ground truths."""
    qformer.eval()
    regressor.eval()

    all_preds, all_gts = [], []
    rows = []

    for batch in tqdm(dataloader, desc="Evaluating"):
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
    parser = argparse.ArgumentParser(description="Evaluate baseline Q-Former checkpoint")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help="Path to the .pth checkpoint file")
    parser.add_argument("--train_csv", type=str, default=DEFAULT_TRAIN_CSV,
                        help="Path to the train CSV file")
    parser.add_argument("--val_csv", type=str, default=DEFAULT_VAL_CSV,
                        help="Path to the val CSV file")
    parser.add_argument("--test_csv", type=str, default=DEFAULT_TEST_CSV,
                        help="Path to the test CSV file")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    # --- Device ---
    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    print(f"Device: {device}")

    # --- Load models ---
    print("Loading Q-Former model...")
    qformer   = QformerWrapper(device=device, is_eval=True).to(device)
    regressor = Regressor(input_dim=768, output_dim=1).to(device)

    # --- Load checkpoint ---
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    qformer.model.Qformer.load_state_dict(ckpt["qformer.Qformer"])
    qformer.model.query_tokens = nn.Parameter(ckpt["query_tokens"].to(device))
    regressor.load_state_dict(ckpt["regressor"])
    print("Checkpoint loaded successfully.")

    # --- Build dataloaders for all splits ---
    splits = {
        "Train": args.train_csv,
        "Val":   args.val_csv,
        "Test":  args.test_csv,
    }

    results = {}  # split_name -> (srcc, plcc, acc_dict)

    for split_name, csv_path in splits.items():
        dataset = QFormerEmbeddingDataset(csv_path=csv_path)
        loader  = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        print(f"\n{split_name} samples: {len(dataset)}")

        all_preds, all_gts, _ = evaluate(qformer, regressor, loader, device)

        srcc = spearmanr_numpy(all_preds, all_gts)
        plcc = pearsonr_numpy(all_preds, all_gts)
        acc  = accuracy_at_threshold(all_preds, all_gts, thresholds=(5.0, 10.0, 15.0))
        results[split_name] = (srcc, plcc, acc)

    # --- Print summary table ---
    thresholds = [5.0, 10.0, 15.0]
    header = f"{'Split':>8} {'SRCC':>10} {'PLCC':>10}"
    for t in thresholds:
        header += f" {'Acc(±' + str(int(t)) + ')':>10}"

    print("\n" + "=" * len(header))
    print("  EVALUATION RESULTS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for split_name, (srcc, plcc, acc) in results.items():
        row = f"{split_name:>8} {srcc:>10.6f} {plcc:>10.6f}"
        for t in thresholds:
            row += f" {acc[t]:>9.2f}%"
        print(row)

    print("=" * len(header))


if __name__ == "__main__":
    main()
