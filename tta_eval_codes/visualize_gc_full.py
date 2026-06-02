"""
Full-dataset GC TTA visualization on A20K.

Runs GC TTA (layernorm, lr=1e-4, 3 steps, batch_size=8) on every batch
in the A20K test set, recording per-image predictions at baseline and
after each TTA step. Generates publication-quality plots.
"""

import os, sys, random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_tta import (
    QformerWrapper, Regressor, ProjectionHead, TTADataset, collate_fn,
    DATASET_CONFIGS, DEFAULT_CHECKPOINT, set_seed, spearmanr, pearsonr,
)
from tta_framework.param_strategy import get_tta_params, freeze_all_except
from tta_framework.losses import GCLoss

# ── Config ─────────────────────────────────────────────────────────────────
DATASET    = "a20k"
SPLIT      = "test"
BATCH_SIZE = 8
TTA_STEPS  = 3
TTA_LR     = 1e-2
STRATEGY   = "layernorm"
GC_P       = 0.25
GC_TEMP    = 0.5
OUT_DIR    = "results/gc_full_viz_lr1e2"

# ── Helpers ────────────────────────────────────────────────────────────────
def _rankdata(a):
    a = np.asarray(a); s = np.argsort(a); inv = np.empty_like(s)
    inv[s] = np.arange(len(a)); a_s = a[s]
    obs = np.concatenate(([True], a_s[1:] != a_s[:-1]))
    dr = np.cumsum(obs); c = np.cumsum(np.bincount(dr))
    return ((c[dr] + c[dr-1] + 1) / 2.0)[inv]


def main():
    set_seed(1234)
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cfg = DATASET_CONFIGS[DATASET]
    csv_path = cfg["splits"][SPLIT]

    print(f"Loading dataset: {DATASET} / {SPLIT}")
    ds = TTADataset(csv_path, cfg, load_raw_images=False)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=collate_fn, num_workers=4, pin_memory=True)

    print(f"Loading checkpoint: {DEFAULT_CHECKPOINT}")
    ckpt = torch.load(DEFAULT_CHECKPOINT, map_location="cpu", weights_only=False)

    qformer   = QformerWrapper(device, is_eval=False).to(device)
    regressor = Regressor(input_dim=768, output_dim=1).to(device)
    proj_head = ProjectionHead(input_dim=768, output_dim=128).to(device)

    qformer.model.Qformer.load_state_dict(ckpt["qformer.Qformer"], strict=True)
    qformer.model.query_tokens = nn.Parameter(ckpt["query_tokens"].to(device))
    regressor.load_state_dict(ckpt["regressor"], strict=True)

    # Snapshot original state for per-batch reset
    orig_query_tokens = qformer.model.query_tokens.detach().clone()
    ln_modules = [m for m in qformer.model.Qformer.modules() if isinstance(m, nn.LayerNorm)]
    orig_ln_states = [
        {"weight": m.weight.detach().clone(), "bias": m.bias.detach().clone()}
        for m in ln_modules
    ]

    gc_loss_fn = GCLoss(p=GC_P, temperature=GC_TEMP)

    # Storage: per-image results at each stage
    # columns: image_name, gt_score, pred_baseline, pred_step1, pred_step2, pred_step3
    all_rows = []

    print(f"\nRunning GC TTA: strategy={STRATEGY}, lr={TTA_LR}, steps={TTA_STEPS}, bs={BATCH_SIZE}")
    print(f"Total batches: {len(loader)}\n")

    for batch_idx, batch in enumerate(tqdm(loader, desc="GC TTA")):
        image_embeds = batch["image_embeds"].to(device)
        prompts = batch["prompts"]
        descs   = batch["descs"]
        gt_scores = batch["gt_scores"].numpy()
        names   = batch["image_names"]
        B = image_embeds.size(0)

        # ── Reset model to checkpoint ──
        with torch.no_grad():
            qformer.model.query_tokens.copy_(orig_query_tokens)
            for m, state in zip(ln_modules, orig_ln_states):
                m.weight.copy_(state["weight"])
                m.bias.copy_(state["bias"])

        # ── Baseline prediction ──
        qformer.eval(); regressor.eval()
        with torch.no_grad():
            mm = qformer.forward_qformer(image_embeds, prompts, descs)
            baseline_preds = regressor(mm).squeeze(-1).cpu().numpy()

        # ── Prepare per-image row storage ──
        batch_rows = []
        for i in range(B):
            batch_rows.append({
                "image_name": names[i],
                "gt_score": gt_scores[i],
                "pred_baseline": baseline_preds[i],
            })

        # ── TTA loop ──
        # Freeze everything, unfreeze layernorms
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
            preds_for_cluster = regressor(mm).squeeze(-1).detach()

            ctx = {
                "proj_feats": proj_feats,
                "predictions": preds_for_cluster,
                "device": device,
            }
            loss = gc_loss_fn(ctx)

            if loss.item() > 0:
                loss.backward()
                optimizer.step()

            # Record predictions after this step
            qformer.eval()
            with torch.no_grad():
                mm_post = qformer.forward_qformer(image_embeds, prompts, descs)
                step_preds = regressor(mm_post).squeeze(-1).cpu().numpy()
            qformer.train()

            for i in range(B):
                batch_rows[i][f"pred_step{step}"] = step_preds[i]

        all_rows.extend(batch_rows)

    # ── Save full results CSV ──
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(OUT_DIR, "full_results.csv"), index=False)
    print(f"\nSaved per-image results: {len(df)} images")

    # ── Compute metrics at each stage ──
    stages = ["pred_baseline", "pred_step1", "pred_step2", "pred_step3"]
    stage_labels = ["Baseline", "Step 1", "Step 2", "Step 3"]
    metrics = []
    for stage, label in zip(stages, stage_labels):
        srcc = spearmanr(df[stage].values, df["gt_score"].values)
        plcc = pearsonr(df[stage].values, df["gt_score"].values)
        metrics.append({"stage": label, "srcc": srcc, "plcc": plcc})
        print(f"  {label:>10s}: SRCC={srcc:.6f}  PLCC={plcc:.6f}")

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(OUT_DIR, "metrics_per_step.csv"), index=False)

    # ══════════════════════════════════════════════════════════════════════
    #  PLOTS
    # ══════════════════════════════════════════════════════════════════════

    # --- Plot 1: SRCC/PLCC progression across TTA steps ---
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(stage_labels))
    srcc_vals = [m["srcc"] for m in metrics]
    plcc_vals = [m["plcc"] for m in metrics]
    ax.plot(x, srcc_vals, "o-", color="#2563EB", linewidth=2.5, markersize=10, label="SRCC")
    ax.plot(x, plcc_vals, "s--", color="#DC2626", linewidth=2.5, markersize=10, label="PLCC")
    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels, fontsize=12)
    ax.set_ylabel("Correlation", fontsize=13)
    ax.set_title("GC TTA: SRCC/PLCC Across Adaptation Steps (A20K Test)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(min(min(srcc_vals), min(plcc_vals)) - 0.01,
                max(max(srcc_vals), max(plcc_vals)) + 0.01)
    for i, (s, p) in enumerate(zip(srcc_vals, plcc_vals)):
        ax.annotate(f"{s:.4f}", (x[i], s), textcoords="offset points", xytext=(0, 12), fontsize=10, color="#2563EB", ha="center")
        ax.annotate(f"{p:.4f}", (x[i], p), textcoords="offset points", xytext=(0, -18), fontsize=10, color="#DC2626", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "metric_progression.png"), dpi=200)
    plt.close()

    # --- Plot 2: Scatter GT vs Pred — Baseline vs Final (side by side) ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, stage, label in zip(axes, ["pred_baseline", "pred_step3"], ["Baseline", "After 3 TTA Steps"]):
        ax.scatter(df["gt_score"], df[stage], alpha=0.15, s=8, color="#6366F1")
        lims = [min(df["gt_score"].min(), df[stage].min()) - 0.05,
                max(df["gt_score"].max(), df[stage].max()) + 0.05]
        ax.plot(lims, lims, "r--", alpha=0.6, label="Ideal (y=x)")
        srcc = spearmanr(df[stage].values, df["gt_score"].values)
        plcc = pearsonr(df[stage].values, df["gt_score"].values)
        ax.set_title(f"{label}\nSRCC={srcc:.4f}  PLCC={plcc:.4f}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Ground Truth Score", fontsize=11)
        ax.set_ylabel("Predicted Score", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.2)
    plt.suptitle("GC TTA on AIGIQA-20K (LayerNorm, LR=1e-4, 3 Steps, BS=8)", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "scatter_baseline_vs_tta.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # --- Plot 3: Per-image error change histogram ---
    df["err_baseline"] = np.abs(df["pred_baseline"] - df["gt_score"])
    df["err_step3"]    = np.abs(df["pred_step3"] - df["gt_score"])
    df["err_change"]   = df["err_step3"] - df["err_baseline"]  # negative = improved

    improved = (df["err_change"] < 0).sum()
    worsened = (df["err_change"] > 0).sum()
    unchanged = (df["err_change"] == 0).sum()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["err_change"], bins=80, color="#8B5CF6", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="No Change")
    ax.axvline(df["err_change"].mean(), color="#059669", linestyle="-", linewidth=2,
               label=f"Mean Δ = {df['err_change'].mean():.4f}")
    ax.set_xlabel("Error Change (negative = improved)", fontsize=12)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_title(f"Per-Image Error Change After GC TTA\n"
                 f"Improved: {improved} | Worsened: {worsened} | Unchanged: {unchanged}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "error_change_histogram.png"), dpi=200)
    plt.close()

    # --- Plot 4: Prediction shift per image (sorted by GT) ---
    df_sorted = df.sort_values("gt_score").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(14, 5))
    x_idx = np.arange(len(df_sorted))
    ax.scatter(x_idx, df_sorted["pred_baseline"], s=3, alpha=0.4, color="#93C5FD", label="Baseline Pred")
    ax.scatter(x_idx, df_sorted["pred_step3"], s=3, alpha=0.4, color="#F87171", label="After TTA Pred")
    ax.plot(x_idx, df_sorted["gt_score"], color="#059669", linewidth=1.2, alpha=0.7, label="Ground Truth")
    ax.set_xlabel("Images (sorted by GT score)", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Prediction Shift: Baseline → TTA (sorted by Ground Truth)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, markerscale=4)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "prediction_shift_sorted.png"), dpi=200)
    plt.close()

    # --- Plot 5: Score movement arrows for random 50 images ---
    sample_df = df.sample(n=min(50, len(df)), random_state=42).sort_values("gt_score").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, row in sample_df.iterrows():
        gt = row["gt_score"]
        p0 = row["pred_baseline"]
        p3 = row["pred_step3"]
        color = "#059669" if abs(p3 - gt) < abs(p0 - gt) else "#DC2626"
        ax.annotate("", xy=(gt, p3), xytext=(gt, p0),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2, alpha=0.6))
    ax.scatter(sample_df["gt_score"], sample_df["pred_baseline"], s=30, color="#2563EB", zorder=5, label="Baseline")
    ax.scatter(sample_df["gt_score"], sample_df["pred_step3"], s=30, color="#DC2626", zorder=5, marker="^", label="After TTA")
    lims = [min(sample_df["gt_score"].min(), sample_df["pred_baseline"].min()) - 0.05,
            max(sample_df["gt_score"].max(), sample_df["pred_baseline"].max()) + 0.05]
    ax.plot(lims, lims, "k--", alpha=0.3, label="Ideal (y=x)")
    ax.set_xlabel("Ground Truth Score", fontsize=12)
    ax.set_ylabel("Predicted Score", fontsize=12)
    ax.set_title("Score Movement Arrows (Green=Improved, Red=Worsened)\n50 Random Images",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "score_movement_arrows.png"), dpi=200)
    plt.close()

    print(f"\nAll plots saved to {OUT_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
