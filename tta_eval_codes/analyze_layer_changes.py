import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Re-use utilities from finetune_and_eval
from finetune_and_eval import (
    set_seed, DATASET_CONFIGS, QFormerEmbeddingDataset, collate_fn,
    QformerWrapper, detect_regressor_type, create_regressor, train_one_epoch
)

def run_analysis(dataset_name, checkpoint_path, epochs=15, fraction=0.2, seed=1234, batch_size=16):
    set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    cfg = DATASET_CONFIGS[dataset_name]
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reg_type = detect_regressor_type(ckpt)
    
    qformer = QformerWrapper(device=device, is_eval=False).to(device)
    regressor = create_regressor(reg_type, ckpt).to(device)
    
    qformer.model.Qformer.load_state_dict(ckpt["qformer.Qformer"], strict=True)
    qformer.model.query_tokens = nn.Parameter(ckpt["query_tokens"].to(device))
    regressor.load_state_dict(ckpt["regressor"], strict=True)
    
    # Freeze/unfreeze
    for p in qformer.model.parameters(): p.requires_grad = False
    qformer.model.query_tokens.requires_grad = True
    for p in qformer.model.Qformer.parameters(): p.requires_grad = True
    for p in regressor.parameters(): p.requires_grad = True

    # Get original weights
    orig_weights = {}
    for name, p in qformer.model.named_parameters():
        if p.requires_grad:
            orig_weights[f"qformer.{name}"] = p.clone().detach().cpu()
    for name, p in regressor.named_parameters():
        if p.requires_grad:
            orig_weights[f"regressor.{name}"] = p.clone().detach().cpu()
            
    # Load dataset
    train_df_full = pd.read_csv(cfg["train_csv"])
    train_df_full.columns = train_df_full.columns.str.strip()
    train_df_sampled = train_df_full.sample(frac=fraction, random_state=seed).reset_index(drop=True)
    train_dataset = QFormerEmbeddingDataset(df=train_df_sampled, embed_root=cfg["embed_root"], embed_format=cfg["embed_format"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        [p for p in qformer.model.parameters() if p.requires_grad] + list(regressor.parameters()), lr=1e-4
    )
    
    print(f"  Training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        train_one_epoch(qformer, regressor, train_loader, optimizer, criterion, device)
        
    # Get trained weights and compute difference
    trained_weights = {}
    for name, p in qformer.model.named_parameters():
        if p.requires_grad:
            trained_weights[f"qformer.{name}"] = p.clone().detach().cpu()
    for name, p in regressor.named_parameters():
        if p.requires_grad:
            trained_weights[f"regressor.{name}"] = p.clone().detach().cpu()
            
    results = []
    for name in orig_weights.keys():
        W_orig = orig_weights[name]
        W_train = trained_weights[name]
        
        avg_abs_orig = W_orig.abs().mean().item()
        avg_abs_change = (W_train - W_orig).abs().mean().item()
        
        rel_change = avg_abs_change / (avg_abs_orig + 1e-8)
        
        results.append({
            "dataset": dataset_name,
            "checkpoint": os.path.basename(checkpoint_path),
            "layer": name,
            "avg_abs_orig": avg_abs_orig,
            "avg_abs_change": avg_abs_change,
            "rel_change": rel_change
        })
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_csv", type=str, default="layer_changes.csv")
    args = parser.parse_args()
    
    configs = [
        # dataset, checkpoint, best_epoch
        ("a20k", "evalmi_baseline_qf.pth", 8),
        ("a20k", "evalmi_baseline_qf_2.pth", 11),
        ("a3k", "evalmi_baseline_qf.pth", 8),
        ("a3k", "evalmi_baseline_qf_2.pth", 4),
    ]
    
    base_ckpt_dir = "/media/parth/021f75bf-bae8-49ef-86a5-28ca19171835/parth/QFormer_IQA_TTA/checkpoints"
    
    all_res = []
    for ds, ckpt, epochs in configs:
        ckpt_path = os.path.join(base_ckpt_dir, ckpt)
        print(f"\n=======================================================")
        print(f"Running analysis for {ds} with {ckpt} (epochs={epochs})")
        print(f"=======================================================")
        df_res = run_analysis(ds, ckpt_path, epochs=epochs, fraction=0.20)
        all_res.append(df_res)
            
    final_df = pd.concat(all_res, ignore_index=True)
    final_df.to_csv(args.out_csv, index=False)
    print(f"\nSaved all results to {args.out_csv}")
