import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from finetune_and_eval import QformerWrapper, create_regressor, detect_regressor_type, QFormerEmbeddingDataset, collate_fn
from tta_framework.param_strategy import get_tta_params
from torch.utils.data import DataLoader

def plot_predictions(preds, gts, names, step_name, title, output_path):
    plt.figure(figsize=(10, 8))
    plt.scatter(gts, preds, color='blue', alpha=0.7, s=100)
    
    # Annotate points
    for i, name in enumerate(names):
        plt.annotate(name, (gts[i], preds[i]), fontsize=10, alpha=0.9, xytext=(8, 8), textcoords='offset points')
        
    # Plot y=x line
    min_val = min(min(gts), min(preds)) - 0.2
    max_val = max(max(gts), max(preds)) + 0.2
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (y=x)')
    
    plt.xlabel('Ground Truth Quality Score')
    plt.ylabel('Predicted Score')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ckpt_path = '../checkpoints/evalmi_baseline_qf.pth'
    
    # Using AIGIQA-20K test split for the 8 images as it's a standard evaluation set
    _SPLIT_DIR = '../important split files-20260527T062853Z-3-001/important split files'
    csv_path = os.path.join(_SPLIT_DIR, "A20K_new", "A20k_test_full_PT1_normalized.csv")
    embed_root = '/media/parth/021f75bf-bae8-49ef-86a5-28ca19171835/parth/dataset/embeddings/a20k'
    
    print("Loading data...")
    dataset = QFormerEmbeddingDataset(csv_path=csv_path, embed_root=embed_root, embed_format='npz')
    
    # Get exactly 8 diverse images by taking evenly spaced indices
    indices = np.linspace(0, len(dataset)-1, 8, dtype=int)
    small_df = dataset.df.iloc[indices].reset_index(drop=True)
    small_dataset = QFormerEmbeddingDataset(df=small_df, embed_root=embed_root, embed_format='npz')
    
    dataloader = DataLoader(small_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(dataloader))
    
    print("Loading model...")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    reg_type = detect_regressor_type(ckpt)
    
    qformer = QformerWrapper(device, is_eval=False).to(device)
    regressor = create_regressor(reg_type, ckpt).to(device)
    
    qformer.model.Qformer.load_state_dict(ckpt['qformer.Qformer'], strict=True)
    qformer.model.query_tokens = nn.Parameter(ckpt['query_tokens'].to(device))
    regressor.load_state_dict(ckpt['regressor'], strict=True)
    
    print("Setting up TTA parameters...")
    for p in qformer.model.parameters(): p.requires_grad = False
    for p in regressor.parameters(): p.requires_grad = False
    
    # Unfreeze layernorms
    ln_params = get_tta_params(qformer, 'layernorm')
    for p in ln_params: p.requires_grad = True
    
    optimizer = torch.optim.Adam(ln_params, lr=1e-4)
    qformer.train()
    
    image_embeds = batch["image_embeds"].to(device)
    prompts = batch["prompts"]
    descs = batch["descs"]
    gt_scores = batch["gt_scores"].cpu().numpy()
    names = batch["image_names"]
    
    out_dir = 'results/gc_visualization'
    os.makedirs(out_dir, exist_ok=True)
    
    print("\n--- BASELINE PASS ---")
    with torch.no_grad():
        baseline_feats = qformer(image_embeds, prompts, descs)
        baseline_preds = regressor(baseline_feats).squeeze(-1)
        baseline_preds_np = baseline_preds.cpu().numpy()
        
    for i in range(8):
        print(f"{names[i]}: Pred = {baseline_preds_np[i]:.4f}, GT = {gt_scores[i]:.4f}")
        
    plot_predictions(baseline_preds_np, gt_scores, names, 'Baseline', 'Baseline Predictions (Before TTA)', f'{out_dir}/baseline.png')
    
    temperature = 0.5
    for step in range(1, 4):
        optimizer.zero_grad()
        
        feats = qformer(image_embeds, prompts, descs)
        preds = regressor(feats).squeeze(-1)
        
        # --- GC Logic ---
        # p=0.25 -> 8 * 0.25 = 2 images per cluster
        k = 2
        idx = torch.argsort(preds)
        
        low_idx = idx[:k]
        high_idx = idx[-k:]
        
        print(f"\n--- STEP {step} ---")
        print("Low Quality Group (Cluster C_low):")
        for i in low_idx: print(f"  - {names[i]} (Pred: {preds[i].item():.4f})")
        
        print("High Quality Group (Cluster C_high):")
        for i in high_idx: print(f"  - {names[i]} (Pred: {preds[i].item():.4f})")
        
        emb_i = feats[low_idx]
        emb_j = feats[high_idx]
        
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)
        
        representations = torch.cat([z_i, z_j], dim=0)
        sim_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        n_i, n_j = 2, 2
        pos_sim_ij = sim_matrix[:n_i, :n_j]
        mask_ij = (~torch.eye(n_i, n_j, dtype=bool, device=device)).float()
        sim_ij = torch.sum(pos_sim_ij * mask_ij, dim=1) / max(n_j - 1, 1)
        
        pos_sim_ji = sim_matrix[n_i:, n_j:]
        mask_ji = (~torch.eye(n_j, n_i, dtype=bool, device=device)).float()
        sim_ji = torch.sum(pos_sim_ji * mask_ji, dim=1) / max(n_i - 1, 1)
        
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        total = 4
        negatives_mask = torch.ones(total, total, dtype=bool, device=device)
        negatives_mask[:n_i, :n_j] = False
        negatives_mask[n_i:, n_j:] = False
        negatives_mask = negatives_mask.float()
        
        nominator = torch.exp(positives / temperature)
        denominator = negatives_mask * torch.exp(sim_matrix / temperature)
        
        loss_partial = torch.sum(nominator / (nominator + torch.sum(denominator, dim=1) + 1e-8)) / total
        loss = -torch.log(loss_partial + 1e-8)
        
        print(f"GC Loss Computed: {loss.item():.4f}")
        
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            new_feats = qformer(image_embeds, prompts, descs)
            new_preds = regressor(new_feats).squeeze(-1).cpu().numpy()
            
        plot_predictions(new_preds, gt_scores, names, f'Step {step}', f'Predictions After Step {step} TTA (LR=1e-4)', f'{out_dir}/step_{step}.png')

if __name__ == '__main__':
    main()
