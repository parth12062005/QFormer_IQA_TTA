import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_tta import QformerWrapper, Regressor, DATASET_CONFIGS, DEFAULT_CHECKPOINT, set_seed, TTADataset

def main():
    set_seed(1234)
    dataset_name = "a20k"
    cfg = DATASET_CONFIGS[dataset_name]
    csv_path = cfg["splits"]["test"]
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_result", "brackets")
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Load Data and Randomly Sample 600
    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    # Randomly sample 600 images
    df = df.sample(n=min(600, len(df)), random_state=42).reset_index(drop=True)
    
    # Sort by GT Score to assign MOS brackets (top 300, bottom 300)
    df = df.sort_values(by=cfg["gt_col"], ascending=False).reset_index(drop=True)
    df.loc[:299, 'mos_category'] = 'high_human'
    df.loc[300:, 'mos_category'] = 'low_human'
    selected_df = df.copy()
    
    # 2. Get Global Predictions
    print("Loading Q-Former...")
    qformer = QformerWrapper(device, is_eval=True, keep_vit=True).to(device)
    regressor = Regressor(input_dim=768, output_dim=1).to(device)
    
    ckpt = torch.load(DEFAULT_CHECKPOINT, map_location="cpu", weights_only=False)
    qformer.model.Qformer.load_state_dict(ckpt["qformer.Qformer"], strict=True)
    qformer.model.query_tokens = nn.Parameter(ckpt["query_tokens"].to(device))
    regressor.load_state_dict(ckpt["regressor"], strict=True)
    
    qformer.eval()
    regressor.eval()
    
    clip_tf = transforms.Compose([
        transforms.Resize((224,224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466,0.4578275,0.40821073), std=(0.26862954,0.26130258,0.27577711)),
    ])
    
    print("Computing global predictions...")
    global_preds = []
    
    img_lookup = TTADataset._build_lookup(cfg["img_root"], cfg["img_subdir"])
    
    for idx, row in tqdm(selected_df.iterrows(), total=len(selected_df)):
        img_name = str(row[cfg["img_col"]])
        base = img_name.replace(".png", cfg["embed_ext"]).replace(".jpg", cfg["embed_ext"]).replace(".jpeg", cfg["embed_ext"])
        embed_path = os.path.join(cfg["embed_root"], base)
        
        if cfg["embed_load"] == "npy":
            image_embeds = torch.from_numpy(np.load(embed_path)).float().unsqueeze(0).to(device)
        else:
            image_embeds = torch.from_numpy(np.load(embed_path)["embed"]).float().unsqueeze(0).to(device)
            
        prompt = str(row[cfg["prompt_col"]]) if cfg.get("prompt_col") else ""
        desc = str(row.get(cfg["desc_col"], "")) if cfg.get("desc_col") else ""
        
        with torch.no_grad():
            mm = qformer.forward_qformer(image_embeds, [prompt], [desc])
            pred = regressor(mm).item()
        global_preds.append(pred)
        
    selected_df['raw_pred'] = global_preds
    
    # Normalize model predictions [0, 1]
    min_pred = selected_df['raw_pred'].min()
    max_pred = selected_df['raw_pred'].max()
    selected_df['norm_pred'] = (selected_df['raw_pred'] - min_pred) / (max_pred - min_pred)
    
    # Sort by prediction to assign Pred brackets (top 300, bottom 300)
    selected_df = selected_df.sort_values(by='raw_pred', ascending=False).reset_index(drop=True)
    selected_df.loc[:299, 'pred_category'] = 'high_pred'
    selected_df.loc[300:, 'pred_category'] = 'low_pred'
    
    filtered_df = selected_df.copy()
    filtered_df['bracket'] = filtered_df['mos_category'] + "_" + filtered_df['pred_category']
    
    for bracket in filtered_df['bracket'].unique():
        os.makedirs(os.path.join(out_dir, bracket), exist_ok=True)
        
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except IOError:
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 22)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 16)
        except IOError:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
    
    print(f"Processing patches for {len(filtered_df)} images in brackets...")
    
    for idx, row in tqdm(filtered_df.iterrows(), total=len(filtered_df)):
        img_name = str(row[cfg["img_col"]])
        
        if img_lookup is None:
            img_path = os.path.join(cfg["img_root"], img_name)
        else:
            img_path = img_lookup.get(img_name)
            
        if img_path is None or not os.path.exists(img_path):
            continue
            
        orig_img = Image.open(img_path).convert("RGB")
        W, H = orig_img.size
        
        patch_w = W // 4
        patch_h = H // 4
        
        prompt = str(row[cfg["prompt_col"]]) if cfg.get("prompt_col") else ""
        desc = str(row.get(cfg["desc_col"], "")) if cfg.get("desc_col") else ""
        
        patch_tensors = []
        coords = []
        
        # 16 patches
        for j in range(4): # row
            for i in range(4): # col
                left = i * patch_w
                upper = j * patch_h
                right = (i + 1) * patch_w if i < 3 else W
                lower = (j + 1) * patch_h if j < 3 else H
                
                patch = orig_img.crop((left, upper, right, lower))
                patch_tensor = clip_tf(patch)
                patch_tensors.append(patch_tensor)
                coords.append((left, upper, right, lower))
                
        # Multiscale wholes
        ms_scales = [1, 2, 4, 8]
        ms_tensors = []
        for scale in ms_scales:
            scaled_pil = orig_img.resize((max(W // scale, 8), max(H // scale, 8)), Image.BICUBIC)
            ms_tensors.append(clip_tf(scaled_pil))
            
        all_tensors = torch.stack(patch_tensors + ms_tensors).to(device)
        
        with torch.no_grad():
            all_embeds = qformer.extract_image_embeds(all_tensors)
            mm = qformer.forward_qformer(all_embeds, [prompt]*len(all_tensors), [desc]*len(all_tensors))
            all_preds = regressor(mm).squeeze(-1).cpu().numpy()
            
        patch_preds = all_preds[:16]
        ms_preds_raw = all_preds[16:]
        ms_preds_norm = [(p - min_pred) / (max_pred - min_pred) for p in ms_preds_raw]
            
        new_width = W + 400
        vis_img = Image.new("RGB", (new_width, max(H, 350)), "white")
        vis_img.paste(orig_img, (0, 0))
        draw = ImageDraw.Draw(vis_img)
        
        for idx_p, (left, upper, right, lower) in enumerate(coords):
            draw.rectangle([left, upper, right, lower], outline="red", width=2)
            
            p_score = patch_preds[idx_p]
            p_score_norm = (p_score - min_pred) / (max_pred - min_pred)
            
            text = f"R:{p_score:.3f}\nN:{p_score_norm:.3f}"
            
            text_bbox = draw.textbbox((left + 5, upper + 5), text, font=font_small)
            draw.rectangle(text_bbox, fill="black")
            draw.text((left + 5, upper + 5), text, fill="white", font=font_small)
            
        overall_text = (
            f"Image: {img_name}\n\n"
            f"Human MOS: {row[cfg['gt_col']]:.4f}\n"
            f"Overall Pred (Raw): {row['raw_pred']:.4f}\n"
            f"Overall Pred (Norm): {row['norm_pred']:.4f}\n\n"
            f"Bracket: {row['bracket']}\n\n"
            f"Multiscale Preds (Raw | Norm):\n"
            f"  1x scale:   {ms_preds_raw[0]:.3f} | {ms_preds_norm[0]:.3f}\n"
            f"  1/2x scale: {ms_preds_raw[1]:.3f} | {ms_preds_norm[1]:.3f}\n"
            f"  1/4x scale: {ms_preds_raw[2]:.3f} | {ms_preds_norm[2]:.3f}\n"
            f"  1/8x scale: {ms_preds_raw[3]:.3f} | {ms_preds_norm[3]:.3f}"
        )
        draw.text((W + 20, 20), overall_text, fill="black", font=font_large)
        
        out_filepath = os.path.join(out_dir, row['bracket'], f"{img_name.replace('/', '_')}")
        vis_img.save(out_filepath)
        
if __name__ == "__main__":
    main()
