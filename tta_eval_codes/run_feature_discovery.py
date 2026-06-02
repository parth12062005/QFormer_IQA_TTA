"""
Orchestrator script for AGIQA Feature Discovery.
Runs the full pipeline: loads images, extracts features, computes correlations,
generates combos, scores for TTA suitability, and produces plots/reports.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
import matplotlib
matplotlib.use("Agg")

# Import feature discovery modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_discovery import FEATURE_REGISTRY, FEATURE_META, register_feature
from feature_discovery.frequency_features import compute_frequency_features, FREQUENCY_FEATURE_META
from feature_discovery.nis_features import compute_nis_features, NIS_FEATURE_META
from feature_discovery.deep_features import (
    compute_deep_features_from_embed, compute_clip_features, compute_dino_features, DEEP_FEATURE_META
)
from feature_discovery.patch_features import compute_patch_features, PATCH_FEATURE_META
from feature_discovery.info_features import compute_info_features, INFO_FEATURE_META
from feature_discovery.multiscale_features import compute_multiscale_features, MULTISCALE_FEATURE_META
from feature_discovery.artifact_features import compute_artifact_features, ARTIFACT_FEATURE_META
from feature_discovery.correlations import compute_all_correlations, split_stability
from feature_discovery.combo_features import generate_combo_features
from feature_discovery.tta_scoring import score_tta_suitability, generate_tta_recommendation
from feature_discovery.plotting import (
    plot_correlation_bar, plot_correlation_heatmap, plot_tta_suitability, plot_top_features_grid
)

# Import dataset infra
from evaluate_tta import DATASET_CONFIGS, TTADataset, collate_fn
from torch.utils.data import DataLoader

# Register all features to the main registry
for k, v in FREQUENCY_FEATURE_META.items(): FEATURE_META[k] = v
for k, v in NIS_FEATURE_META.items(): FEATURE_META[k] = v
for k, v in DEEP_FEATURE_META.items(): FEATURE_META[k] = v
for k, v in PATCH_FEATURE_META.items(): FEATURE_META[k] = v
for k, v in INFO_FEATURE_META.items(): FEATURE_META[k] = v
for k, v in MULTISCALE_FEATURE_META.items(): FEATURE_META[k] = v
for k, v in ARTIFACT_FEATURE_META.items(): FEATURE_META[k] = v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="a20k")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="feature_discovery_results")
    parser.add_argument("--splits", nargs="+", default=["test", "val"], help="Dataset splits to use")
    parser.add_argument("--skip_deep_models", action="store_true", help="Skip loading CLIP/DINO to save memory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plot_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load CLIP and DINO if not skipped
    clip_model, clip_preprocess = None, None
    dino_model = None
    if not args.skip_deep_models:
        try:
            import clip
            print("Loading CLIP (ViT-B/32)...")
            clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
            clip_model.eval()
        except ImportError:
            print("Warning: CLIP not installed. Skipping CLIP features.")
            
        try:
            import timm
            print("Loading DINO (vit_small_patch16_224.dino)...")
            dino_model = timm.create_model("vit_small_patch16_224.dino", pretrained=True).to(device)
            dino_model.eval()
        except Exception as e:
            print(f"Warning: DINO loading failed ({e}). Skipping DINO features.")

    # Datasets
    cfg = DATASET_CONFIGS[args.dataset]
    all_data = []

    for split in args.splits:
        print(f"\nProcessing {args.dataset} split: {split}")
        csv_path = cfg["splits"][split]
        ds = TTADataset(csv_path, cfg, load_raw_images=True)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
        
        for batch in tqdm(loader, desc=f"Extracting features ({split})"):
            embeds = batch["image_embeds"].numpy() # (B, 257, 1408)
            mos_scores = batch["gt_scores"].numpy()
            img_names = batch["image_names"]
            
            # Access raw images via the TTADataset lookup logic (copied from evaluate_tta.py structure)
            for i in range(len(img_names)):
                img_name = img_names[i]
                mos = mos_scores[i]
                emb = embeds[i]
                
                # Get image path
                if ds.img_lookup is None:
                    img_path = os.path.join(cfg["img_root"], img_name)
                else:
                    img_path = ds.img_lookup.get(img_name)
                
                if not img_path or not os.path.exists(img_path):
                    continue
                    
                img_pil = Image.open(img_path).convert("RGB")
                img_np = np.array(img_pil)
                
                # Container for this image's features
                feat_dict = {"image_name": img_name, "split": split, "mos": mos}
                
                # 1. Frequency features
                feat_dict.update(compute_frequency_features(img_np))
                # 2. NIS features
                feat_dict.update(compute_nis_features(img_np))
                # 3. Patch features
                feat_dict.update(compute_patch_features(img_np))
                # 4. Info features
                feat_dict.update(compute_info_features(img_np))
                # 5. Multiscale features
                feat_dict.update(compute_multiscale_features(img_np))
                # 6. Artifact features
                feat_dict.update(compute_artifact_features(img_np))
                # 7. Deep features (ViT embeddings)
                feat_dict.update(compute_deep_features_from_embed(emb))
                
                # 8. Deep features (CLIP / DINO)
                if clip_model is not None:
                    c_input = clip_preprocess(img_pil).unsqueeze(0)
                    feat_dict.update(compute_clip_features(c_input, clip_model, clip_preprocess, device))
                if dino_model is not None:
                    from torchvision import transforms
                    tfm = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                    ])
                    d_input = tfm(img_pil).unsqueeze(0)
                    feat_dict.update(compute_dino_features(d_input, dino_model, device))
                    
                all_data.append(feat_dict)

    # ── Compilation & Analysis ──
    print("\nFeature extraction complete. Compiling results...")
    df = pd.DataFrame(all_data)
    df.to_csv(os.path.join(args.output_dir, "all_features_base.csv"), index=False)
    
    base_feature_cols = [c for c in df.columns if c not in ["image_name", "split", "mos"]]
    mos_values = df["mos"].values

    # Correlation Analysis
    print("Computing correlations...")
    corr_results = []
    for feat in tqdm(base_feature_cols, desc="Correlations"):
        f_vals = df[feat].values
        corrs = compute_all_correlations(f_vals, mos_values)
        stability = split_stability(f_vals, mos_values)
        
        row = {"feature": feat}
        row.update(corrs)
        row.update(stability)
        row["abs_spearman"] = abs(corrs["spearman"])
        
        # TTA Scoring
        row["tta_score"] = score_tta_suitability(feat, row["abs_spearman"], row["spearman_std"], FEATURE_META)
        corr_results.append(row)

    rank_df = pd.DataFrame(corr_results)
    rank_df = rank_df.sort_values("abs_spearman", ascending=False)
    rank_df.to_csv(os.path.join(args.output_dir, "correlation_rankings_base.csv"), index=False)

    # Combinations
    print("\nGenerating combo features from Top-20 base features...")
    combo_df = generate_combo_features(df, base_feature_cols, rank_df, top_k=20)
    
    combo_corr_results = []
    for feat in tqdm(combo_df.columns, desc="Combo Correlations"):
        f_vals = combo_df[feat].values
        corrs = compute_all_correlations(f_vals, mos_values)
        row = {"feature": feat}
        row.update(corrs)
        row["abs_spearman"] = abs(corrs["spearman"])
        combo_corr_results.append(row)
        
    combo_rank_df = pd.DataFrame(combo_corr_results).sort_values("abs_spearman", ascending=False)
    combo_rank_df.to_csv(os.path.join(args.output_dir, "correlation_rankings_combo.csv"), index=False)

    # Combine everything for final output
    full_df = pd.concat([df, combo_df], axis=1)
    full_df.to_csv(os.path.join(args.output_dir, "all_features_full.csv"), index=False)

    # Plotting
    print("\nGenerating plots...")
    plot_correlation_bar(rank_df, top_n=30, metric="abs_spearman", 
                         save_path=os.path.join(plot_dir, "top30_spearman_base.png"))
    plot_correlation_bar(combo_rank_df, top_n=30, metric="abs_spearman", 
                         save_path=os.path.join(plot_dir, "top30_spearman_combo.png"))
    
    top40_feats = rank_df.nlargest(40, "abs_spearman")["feature"].tolist()
    plot_correlation_heatmap(df, top40_feats, save_path=os.path.join(plot_dir, "heatmap_top40.png"))
    
    plot_tta_suitability(rank_df, top_n=30, save_path=os.path.join(plot_dir, "tta_suitability.png"))
    
    plot_top_features_grid(df, mos_values, rank_df, top_n=20, save_dir=os.path.join(plot_dir, "top20_scatter_dist"))

    # Report Generation
    print("\nGenerating Report...")
    report_path = os.path.join(args.output_dir, "feature_discovery_report.md")
    with open(report_path, "w") as f:
        f.write("# AGIQA Feature Discovery Report\n\n")
        f.write(f"**Dataset Splits**: {', '.join(args.splits)}\n")
        f.write(f"**Total Images Processed**: {len(df)}\n\n")
        
        f.write("## Top 20 Base Features by Spearman Correlation\n\n")
        f.write("| Rank | Feature | Spearman | Pearson | MI | TTA Score |\n")
        f.write("|---|---|---|---|---|---|\n")
        for i, (_, row) in enumerate(rank_df.head(20).iterrows(), 1):
            f.write(f"| {i} | `{row['feature']}` | {row['spearman']:.4f} | {row['pearson']:.4f} | {row['mutual_info']:.4f} | {row['tta_score']}/10 |\n")
            
        f.write("\n## Top 10 Combo Features\n\n")
        f.write("| Rank | Feature | Spearman | Pearson |\n")
        f.write("|---|---|---|---|\n")
        for i, (_, row) in enumerate(combo_rank_df.head(10).iterrows(), 1):
            f.write(f"| {i} | `{row['feature']}` | {row['spearman']:.4f} | {row['pearson']:.4f} |\n")
            
        f.write("\n## Top TTA Candidates (Score >= 7)\n\n")
        tta_candidates = rank_df[rank_df["tta_score"] >= 7].sort_values("tta_score", ascending=False)
        for _, row in tta_candidates.iterrows():
            feat = row["feature"]
            score = row["tta_score"]
            meta = FEATURE_META.get(feat, {})
            f.write(f"### `{feat}` (Score: {score}/10)\n")
            f.write(f"- **Description**: {meta.get('description', 'N/A')}\n")
            f.write(f"- **Spearman vs MOS**: {row['spearman']:.4f}\n")
            f.write(f"- **Recommendation**: {generate_tta_recommendation(feat, meta)}\n\n")

    print(f"\nDone! All results saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
