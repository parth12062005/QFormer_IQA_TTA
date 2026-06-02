import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_tta import QformerWrapper, Regressor, ProjectionHead, TTADataset, collate_fn, DATASET_CONFIGS, DEFAULT_CHECKPOINT, set_seed
from tta_framework.param_strategy import get_tta_params, freeze_all_except
from tta_framework.losses import GCLoss

def _get_ranks(scores):
    # rank 1 for highest score, rank N for lowest
    order = np.argsort(-np.array(scores))
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks

def draw_text_on_image(img_path, out_path, mos_score, mos_rank, base_score, base_rank, tta_score, tta_rank):
    img = Image.open(img_path).convert("RGB")
    width, height = img.size
    
    # Add padding to the right for text
    pad_width = 350
    new_width = width + pad_width
    new_img = Image.new("RGB", (new_width, max(height, 250)), "white")
    new_img.paste(img, (0, 0))
    
    draw = ImageDraw.Draw(new_img)
    
    try:
        # Try a standard system font
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except IOError:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 22)
        except IOError:
            font = ImageFont.load_default()
        
    text = (
        f"MOS Score: {mos_score:.4f}\n"
        f"MOS Rank: {mos_rank}\n\n"
        f"Baseline Score: {base_score:.4f}\n"
        f"Baseline Rank: {base_rank}\n\n"
        f"TTA Score: {tta_score:.4f}\n"
        f"TTA Rank: {tta_rank}"
    )
            
    draw.text((width + 20, 20), text, fill="black", font=font)
    new_img.save(out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="a20k")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--tta_lr", type=float, default=0.001)
    parser.add_argument("--tta_steps", type=int, default=3)
    parser.add_argument("--strategy", type=str, default="layernorm")
    parser.add_argument("--max_batches", type=int, default=10, help="Max batches to visualize")
    args = parser.parse_args()

    set_seed(1234)
    out_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_result")
    os.makedirs(out_root, exist_ok=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = DATASET_CONFIGS[args.dataset]
    csv_path = cfg["splits"][args.split]
    
    ds = TTADataset(csv_path, cfg, load_raw_images=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    ckpt = torch.load(DEFAULT_CHECKPOINT, map_location="cpu", weights_only=False)
    qformer = QformerWrapper(device, is_eval=False).to(device)
    regressor = Regressor(input_dim=768, output_dim=1).to(device)
    proj_head = ProjectionHead(input_dim=768, output_dim=128).to(device)
    
    qformer.model.Qformer.load_state_dict(ckpt["qformer.Qformer"], strict=True)
    qformer.model.query_tokens = nn.Parameter(ckpt["query_tokens"].to(device))
    regressor.load_state_dict(ckpt["regressor"], strict=True)
    
    orig_query_tokens = qformer.model.query_tokens.detach().clone()
    ln_modules = [m for m in qformer.model.Qformer.modules() if isinstance(m, nn.LayerNorm)]
    orig_ln_states = [{"weight": m.weight.detach().clone(), "bias": m.bias.detach().clone()} for m in ln_modules]
    
    gc_loss_fn = GCLoss(p=0.25, temperature=0.5)
    img_root = cfg["img_root"]
    lookup = TTADataset._build_lookup(cfg["img_root"], cfg["img_subdir"])
    
    for batch_idx, batch in enumerate(tqdm(loader, desc="Visualizing Batches")):
        if batch_idx >= args.max_batches:
            break
            
        batch_folder = os.path.join(out_root, f"batch{batch_idx+1}")
        os.makedirs(batch_folder, exist_ok=True)
        
        image_embeds = batch["image_embeds"].to(device)
        prompts = batch["prompts"]
        descs = batch["descs"]
        gt_scores = batch["gt_scores"].numpy()
        names = batch["image_names"]
        B = image_embeds.size(0)
        
        # Reset model
        with torch.no_grad():
            qformer.model.query_tokens.copy_(orig_query_tokens)
            for m, state in zip(ln_modules, orig_ln_states):
                m.weight.copy_(state["weight"])
                m.bias.copy_(state["bias"])
                
        # Baseline pred
        qformer.eval(); regressor.eval()
        with torch.no_grad():
            mm = qformer.forward_qformer(image_embeds, prompts, descs)
            baseline_preds = regressor(mm).squeeze(-1).cpu().numpy()
            
        # TTA
        for p in qformer.model.parameters(): p.requires_grad = False
        for p in regressor.parameters(): p.requires_grad = False
        for p in proj_head.parameters(): p.requires_grad = False
        
        ln_params = get_tta_params(qformer, args.strategy)
        freeze_all_except(qformer, ln_params)
        
        optimizer = torch.optim.Adam(ln_params, lr=args.tta_lr)
        qformer.train()
        
        for step in range(args.tta_steps):
            optimizer.zero_grad()
            mm = qformer.forward_qformer(image_embeds, prompts, descs)
            proj_feats = proj_head(mm)
            preds_cluster = regressor(mm).squeeze(-1).detach()
            ctx = {"proj_feats": proj_feats, "predictions": preds_cluster, "device": device}
            loss = gc_loss_fn(ctx)
            if loss.item() > 0:
                loss.backward()
                optimizer.step()
                
        qformer.eval()
        with torch.no_grad():
            mm_post = qformer.forward_qformer(image_embeds, prompts, descs)
            tta_preds = regressor(mm_post).squeeze(-1).cpu().numpy()
            
        # Compute Ranks within the batch
        mos_ranks = _get_ranks(gt_scores)
        base_ranks = _get_ranks(baseline_preds)
        tta_ranks = _get_ranks(tta_preds)
        
        for i in range(B):
            if lookup is None:
                img_path = os.path.join(img_root, names[i])
            else:
                img_path = lookup.get(names[i])
                
            if img_path is None or not os.path.exists(img_path):
                print(f"Warning: Image {names[i]} not found, skipping.")
                continue
                
            out_path = os.path.join(batch_folder, f"{mos_ranks[i]}.jpg")
            draw_text_on_image(img_path, out_path, 
                               gt_scores[i], mos_ranks[i],
                               baseline_preds[i], base_ranks[i],
                               tta_preds[i], tta_ranks[i])

    print(f"Visualizations saved to {out_root}")

if __name__ == "__main__":
    main()
