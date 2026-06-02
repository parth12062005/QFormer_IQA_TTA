import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

INPUT_CSV = "baseline_1000_sample.csv"
IMAGE_DIR = "/media/parth/021f75bf-bae8-49ef-86a5-28ca19171835/parth/agiqa-20k/AIGCQA-30K-Image"
OUTPUT_PDF = "a20k_baseline_evaluation.pdf"

def compute_metrics(gts, preds):
    srcc, _ = spearmanr(gts, preds)
    plcc, _ = pearsonr(gts, preds)
    return srcc, plcc

def main():
    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    assert len(df) == 1000, f"Expected 1000 rows, got {len(df)}"
    
    # Chunk into 125 pages (8 per page)
    groups = [df.iloc[i:i+8] for i in range(0, 1000, 8)]
    
    print(f"Generating PDF with {len(groups)} pages...")
    with PdfPages(OUTPUT_PDF) as pdf:
        for page_idx, group in enumerate(tqdm(groups, desc="Pages")):
            # Rank 1 = Highest score
            group = group.sort_values(by="gt_score", ascending=False).reset_index(drop=True)
            group["gt_rank"] = np.arange(1, 9)
            
            # Predict rank
            group["pred_rank"] = group["pred_score"].rank(ascending=False, method='min').astype(int)
            
            # Metrics for this page
            srcc, plcc = compute_metrics(group["gt_score"].values, group["pred_score"].values)
            
            # Setup figure
            fig, axes = plt.subplots(8, 1, figsize=(8.5, 11))
            fig.subplots_adjust(top=0.92, bottom=0.05, left=0.05, right=0.95, hspace=0.3)
            
            fig.suptitle(f"Set {page_idx + 1}/125 | SRCC: {srcc:.4f} | PLCC: {plcc:.4f}\nModel: EvalMI Baseline (Zero-Shot)", fontsize=14, fontweight='bold')
            
            for i, row in group.iterrows():
                ax = axes[i]
                ax.axis('off')
                
                # Load image
                img_path = None
                for split in ["train", "val", "test"]:
                    candidate = os.path.join(IMAGE_DIR, split, row['image_name'])
                    if os.path.exists(candidate):
                        img_path = candidate
                        break
                
                try:
                    if img_path is None:
                        raise FileNotFoundError
                    img = Image.open(img_path).convert("RGB")
                    ax.imshow(img, aspect='auto', extent=[0, 0.4, 0, 1])
                except Exception as e:
                    ax.text(0.2, 0.5, "Image Missing", ha='center', va='center')
                
                # Add text
                # We draw the image from x=0 to 0.4, so text goes from 0.45 to 1.0
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                
                gt_text = f"GROUND TRUTH\n\nRank: {row['gt_rank']}\nScore: {row['gt_score']:.4f}"
                pred_text = f"PREDICTION\n\nRank: {row['pred_rank']}\nScore: {row['pred_score']:.4f}"
                
                # Use colors to indicate if rank matches
                pred_color = "green" if row['gt_rank'] == row['pred_rank'] else "red"
                
                ax.text(0.45, 0.5, gt_text, ha='left', va='center', fontsize=12, fontweight='bold')
                ax.text(0.75, 0.5, pred_text, ha='left', va='center', fontsize=12, fontweight='bold', color=pred_color)
                
                # Draw a separator line at bottom
                if i < 7:
                    ax.plot([0, 1], [-0.15, -0.15], color='black', lw=1, clip_on=False)
            
            pdf.savefig(fig)
            plt.close(fig)

    print(f"PDF saved to {OUTPUT_PDF}")

if __name__ == "__main__":
    main()
