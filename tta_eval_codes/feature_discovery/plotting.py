"""
Plotting utilities for feature discovery results.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def plot_scatter(feature_values, mos_values, feature_name, pearson, spearman,
                 save_path=None):
    """Scatter plot of feature vs MOS with correlation info."""
    fig, ax = plt.subplots(figsize=(7, 5))
    mask = np.isfinite(feature_values) & np.isfinite(mos_values)
    fv, mv = feature_values[mask], mos_values[mask]
    
    ax.scatter(fv, mv, alpha=0.3, s=8, c="#4C72B0")
    
    # Trend line
    if len(fv) > 2:
        z = np.polyfit(fv, mv, 1)
        p = np.poly1d(z)
        x_line = np.linspace(np.min(fv), np.max(fv), 100)
        ax.plot(x_line, p(x_line), "r-", linewidth=2, alpha=0.7)
    
    ax.set_xlabel(feature_name, fontsize=12)
    ax.set_ylabel("MOS", fontsize=12)
    ax.set_title(f"{feature_name}\nPearson={pearson:.4f}  Spearman={spearman:.4f}",
                 fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_distribution(feature_values, feature_name, save_path=None):
    """Distribution histogram of a feature."""
    fig, ax = plt.subplots(figsize=(7, 4))
    mask = np.isfinite(feature_values)
    fv = feature_values[mask]
    
    ax.hist(fv, bins=50, alpha=0.7, color="#4C72B0", edgecolor="white")
    ax.axvline(np.mean(fv), color="red", linestyle="--", label=f"Mean={np.mean(fv):.4f}")
    ax.axvline(np.median(fv), color="green", linestyle="--", label=f"Median={np.median(fv):.4f}")
    
    ax.set_xlabel(feature_name, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Distribution: {feature_name}", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_bar(rankings_df, top_n=30, metric="abs_spearman",
                         save_path=None):
    """Horizontal bar chart of top features by correlation."""
    df = rankings_df.nlargest(top_n, metric)
    
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    colors = ["#c44e52" if v < 0 else "#4c72b0" 
              for v in df["spearman"]]
    ax.barh(range(len(df)), df[metric].values, color=colors, alpha=0.8)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["feature"].values, fontsize=8)
    ax.set_xlabel(f"|{metric}|", fontsize=12)
    ax.set_title(f"Top {top_n} Features by |Spearman|", fontsize=13)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(feature_df, feature_names, save_path=None):
    """Heatmap of inter-feature correlations."""
    n = len(feature_names)
    if n > 40:
        feature_names = feature_names[:40]
    
    data = feature_df[feature_names].values
    # Remove non-finite values columnwise
    for j in range(data.shape[1]):
        col = data[:, j]
        mask = ~np.isfinite(col)
        if mask.any():
            col[mask] = np.nanmedian(col[~mask]) if np.any(~mask) else 0
    
    corr_matrix = np.corrcoef(data.T)
    
    fig, ax = plt.subplots(figsize=(max(10, n * 0.4), max(8, n * 0.35)))
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(feature_names)))
    ax.set_yticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=90, fontsize=6)
    ax.set_yticklabels(feature_names, fontsize=6)
    ax.set_title("Inter-Feature Correlation Heatmap", fontsize=13)
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_tta_suitability(tta_df, top_n=30, save_path=None):
    """Bar chart of TTA suitability scores."""
    df = tta_df.nlargest(top_n, "tta_score")
    
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    bars = ax.barh(range(len(df)), df["tta_score"].values, 
                   color="#55a868", alpha=0.8)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["feature"].values, fontsize=8)
    ax.set_xlabel("TTA Suitability Score (0–10)", fontsize=12)
    ax.set_title(f"Top {top_n} Features by TTA Suitability", fontsize=13)
    ax.set_xlim(0, 10)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_top_features_grid(feature_df, mos_values, rankings_df,
                           top_n=20, save_dir=None):
    """Generate scatter + distribution plots for top N features."""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    df = rankings_df.nlargest(top_n, "abs_spearman")
    for _, row in df.iterrows():
        fname = row["feature"]
        fvals = feature_df[fname].values if fname in feature_df.columns else None
        if fvals is None:
            continue
        
        pearson = row.get("pearson", 0)
        spearman = row.get("spearman", 0)
        
        if save_dir:
            plot_scatter(fvals, mos_values, fname, pearson, spearman,
                        save_path=os.path.join(save_dir, f"scatter_{fname}.png"))
            plot_distribution(fvals, fname,
                            save_path=os.path.join(save_dir, f"dist_{fname}.png"))
