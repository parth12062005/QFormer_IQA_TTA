import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Re-use utilities from finetune_and_eval
from finetune_and_eval import (
    set_seed, DATASET_CONFIGS, QFormerEmbeddingDataset, collate_fn,
    QformerWrapper, detect_regressor_type, create_regressor
)

def compute_fisher_layer_scores(dataset_name, checkpoint_path, fraction=0.2, seed=1234, batch_size=16):
    set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    cfg = DATASET_CONFIGS[dataset_name]
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reg_type = detect_regressor_type(ckpt)
    
    qformer = QformerWrapper(device=device, is_eval=True).to(device)
    regressor = create_regressor(reg_type, ckpt).to(device)
    
    qformer.model.Qformer.load_state_dict(ckpt["qformer.Qformer"], strict=True)
    qformer.model.query_tokens = nn.Parameter(ckpt["query_tokens"].to(device))
    regressor.load_state_dict(ckpt["regressor"], strict=True)
    
    # We want gradients for all parameters to compute Fisher
    for p in qformer.model.parameters(): p.requires_grad = True
    qformer.model.query_tokens.requires_grad = True
    for p in regressor.parameters(): p.requires_grad = True

    # Initialize Fisher accumulator
    fisher_dict_qf = {name: torch.zeros_like(param) for name, param in qformer.model.named_parameters() if param.requires_grad}
    fisher_dict_reg = {name: torch.zeros_like(param) for name, param in regressor.named_parameters() if param.requires_grad}
    total_samples = 0
            
    # Load dataset
    train_df_full = pd.read_csv(cfg["train_csv"])
    train_df_full.columns = train_df_full.columns.str.strip()
    train_df_sampled = train_df_full.sample(frac=fraction, random_state=seed).reset_index(drop=True)
    train_dataset = QFormerEmbeddingDataset(df=train_df_sampled, embed_root=cfg["embed_root"], embed_format=cfg["embed_format"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    
    loss_fn = nn.MSELoss()
    
    qformer.model.eval()
    regressor.eval()
    
    print(f"  Computing FIM over {len(train_loader)} batches...")
    for step, (images, targets) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        targets = targets.to(device).float().view(-1, 1)
        
        # Forward pass
        qf_output = qformer(images)
        outputs = regressor(qf_output)
        loss = loss_fn(outputs, targets)
        
        # Backward pass
        qformer.model.zero_grad()
        regressor.zero_grad()
        loss.backward()
        
        # Accumulate
        b_size = images.size(0)
        total_samples += b_size
        
        with torch.no_grad():
            for name, param in qformer.model.named_parameters():
                if param.grad is not None:
                    fisher_dict_qf[name] += (param.grad ** 2) * b_size
            for name, param in regressor.named_parameters():
                if param.grad is not None:
                    fisher_dict_reg[name] += (param.grad ** 2) * b_size
        
    # Normalize and aggregate by named structural layers
    layer_scores = {}
    
    def process_fisher_dict(fisher_dict, prefix=""):
        with torch.no_grad():
            for name, fisher_sum in fisher_dict.items():
                # Average across the dataset size
                fisher_normalized = fisher_sum / total_samples
                
                # Determine base layer name 
                # e.g., 'Qformer.bert.encoder.layer.0.attention.self.query.weight' -> 'Qformer.bert.encoder.layer.0.attention.self.query'
                # or just 'Qformer.bert.encoder.layer.0'
                # Let's keep the parameter name to match the previous layer analysis
                full_name = f"{prefix}.{name}" if prefix else name
                
                param_mean_fisher = fisher_normalized.mean().item()
                layer_scores[full_name] = param_mean_fisher

    process_fisher_dict(fisher_dict_qf, "qformer")
    process_fisher_dict(fisher_dict_reg, "regressor")
            
    # Format results
    results = []
    for layer_name, score in layer_scores.items():
        results.append({
            "dataset": dataset_name,
            "checkpoint": os.path.basename(checkpoint_path),
            "layer": layer_name,
            "fisher_score": score
        })
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_csv", type=str, default="fisher_scores.csv")
    args = parser.parse_args()
    
    configs = [
        ("a20k", "evalmi_baseline_qf.pth"),
        ("a20k", "evalmi_baseline_qf_2.pth"),
        ("a3k", "evalmi_baseline_qf.pth"),
        ("a3k", "evalmi_baseline_qf_2.pth"),
    ]
    
    base_ckpt_dir = "/media/parth/021f75bf-bae8-49ef-86a5-28ca19171835/parth/QFormer_IQA_TTA/checkpoints"
    
    all_res = []
    for ds, ckpt in configs:
        ckpt_path = os.path.join(base_ckpt_dir, ckpt)
        print(f"\n=======================================================")
        print(f"Running Fisher analysis for {ds} with {ckpt}")
        print(f"=======================================================")
        df_res = compute_fisher_layer_scores(ds, ckpt_path, fraction=0.20)
        all_res.append(df_res)
            
    final_df = pd.concat(all_res, ignore_index=True)
    final_df.to_csv(args.out_csv, index=False)
    print(f"\nSaved all results to {args.out_csv}")
