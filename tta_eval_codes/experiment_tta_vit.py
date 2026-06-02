import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_tta import (
    QformerWrapper, Regressor, DATASET_CONFIGS,
    DEFAULT_CHECKPOINT, set_seed, spearmanr, pearsonr,
)

# ── Config ─────────────────────────────────────────────────────────────────
DATASET    = "a20k"
SPLIT      = "test"
BATCH_SIZE = 8
TTA_STEPS  = 1
TTA_LR     = 5e-3
OUT_DIR    = "results/exp_tta_vit"

# ── Augmentations ──────────────────────────────────────────────────────────
base_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                         std=(0.26862954, 0.26130258, 0.27577711)),
])

def gaussian_blur(img, sigma):
    import torchvision.transforms.functional as TF
    return TF.gaussian_blur(img, kernel_size=[5, 5], sigma=[sigma, sigma])

def compress_image(img, quality):
    import io
    buf = io.BytesIO()
    img_pil = transforms.ToPILImage()(img) if isinstance(img, torch.Tensor) else img
    img_pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def add_noise(img, var):
    import cv2
    from skimage.util import random_noise
    img_cv = np.array(img)
    noise = random_noise(img_cv, mode='gaussian', var=var)
    return Image.fromarray((noise * 255).astype('uint8'))

class ViT_TTADataset(Dataset):
    def __init__(self, csv_path, cfg):
        self.df = pd.read_csv(csv_path)
        self.df.columns = self.df.columns.str.strip()
        self.cfg = cfg
        
        self.img_lookup = {}
        img_root = cfg["img_root"]
        if cfg.get("img_subdir"):
            for sd in os.listdir(img_root):
                sp = os.path.join(img_root, sd)
                if not os.path.isdir(sp): continue
                for f in os.listdir(sp):
                    if f.lower().endswith(('.png','.jpg','.jpeg')):
                        self.img_lookup[f] = os.path.join(sp, f)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row[self.cfg["img_col"]])
        
        img_path = self.img_lookup.get(img_name)
        pil_img = Image.open(img_path).convert("RGB")
        
        img_base = base_transform(pil_img)
        
        # Rank Augs
        # Comp
        q_low = int(80 + np.random.random() * 10)
        q_high = int(40 + np.random.random() * 20)
        img_comp_low = base_transform(compress_image(pil_img, q_low))
        img_comp_high = base_transform(compress_image(pil_img, q_high))
        
        # Noise
        var_low = 0.00001 + np.random.random() * 0.000001
        var_high = 0.00005 + np.random.random() * 0.000001
        img_nos_low = base_transform(add_noise(pil_img, var_low))
        img_nos_high = base_transform(add_noise(pil_img, var_high))
        
        # Blur
        sig_low = 1 + np.random.random() * 19
        sig_high = 40 + np.random.random() * 40
        img_blur_low = base_transform(gaussian_blur(pil_img, sig_low))
        img_blur_high = base_transform(gaussian_blur(pil_img, sig_high))
        
        prompt = str(row.get(self.cfg.get("prompt_col", "prompt"), ""))
        desc = str(row.get(self.cfg.get("desc_col", "gen_answer"), ""))
        gt = torch.tensor(float(row[self.cfg["gt_col"]]), dtype=torch.float32)
        
        return {
            "img_base": img_base,
            "img_comp_low": img_comp_low,
            "img_comp_high": img_comp_high,
            "img_nos_low": img_nos_low,
            "img_nos_high": img_nos_high,
            "img_blur_low": img_blur_low,
            "img_blur_high": img_blur_high,
            "prompt": prompt,
            "description": desc,
            "image_name": img_name,
            "gt_score": gt,
        }

def vit_collate_fn(batch):
    return {
        "img_base":   torch.stack([b["img_base"] for b in batch]),
        "img_comp_low": torch.stack([b["img_comp_low"] for b in batch]),
        "img_comp_high": torch.stack([b["img_comp_high"] for b in batch]),
        "img_nos_low": torch.stack([b["img_nos_low"] for b in batch]),
        "img_nos_high": torch.stack([b["img_nos_high"] for b in batch]),
        "img_blur_low": torch.stack([b["img_blur_low"] for b in batch]),
        "img_blur_high": torch.stack([b["img_blur_high"] for b in batch]),
        "prompts": [b["prompt"] for b in batch],
        "descs": [b["description"] for b in batch],
        "image_names": [b["image_name"] for b in batch],
        "gt_scores": torch.stack([b["gt_score"] for b in batch]),
    }

# ── Losses ─────────────────────────────────────────────────────────────────
def gc_loss(features, preds, p=0.25, temperature=0.5, device=None):
    B = features.size(0)
    k = max(2, int(B * p))
    if B < 4:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    idx = torch.argsort(preds)
    z_i = F.normalize(features[idx[:k]], dim=1)
    z_j = F.normalize(features[idx[-k:]], dim=1)
    n_i, n_j = z_i.size(0), z_j.size(0)
    
    reps = torch.cat([z_i, z_j], dim=0)
    sim = F.cosine_similarity(reps.unsqueeze(1), reps.unsqueeze(0), dim=2)
    
    ps_ij = sim[:n_i, :n_j]
    m_ij = (~torch.eye(n_i, n_j, dtype=bool, device=device)).float()
    s_ij = torch.sum(ps_ij * m_ij, dim=1) / max(n_j - 1, 1)
    
    ps_ji = sim[n_i:, n_j:]
    m_ji = (~torch.eye(n_j, n_i, dtype=bool, device=device)).float()
    s_ji = torch.sum(ps_ji * m_ji, dim=1) / max(n_i - 1, 1)
    
    pos = torch.cat([s_ij, s_ji], dim=0)
    total = n_i + n_j
    nm = torch.ones(total, total, dtype=bool, device=device)
    nm[:n_i, :n_j] = False; nm[n_i:, n_j:] = False
    nm = nm.float()
    
    nom = torch.exp(pos / temperature)
    den = nm * torch.exp(sim / temperature)
    lp = torch.sum(nom / (nom + torch.sum(den, dim=1) + 1e-8)) / total
    return -torch.log(lp + 1e-8)

def rank_loss(feat_orig, feat_weak, feat_strong):
    d_weak   = F.pairwise_distance(feat_weak, feat_orig, p=2)
    d_strong = F.pairwise_distance(feat_strong, feat_orig, p=2)
    target = torch.ones_like(d_strong)
    return F.binary_cross_entropy_with_logits(d_strong - d_weak, target)

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    set_seed(1234)
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    cfg = DATASET_CONFIGS[DATASET]
    csv_path = cfg["splits"][SPLIT]
    
    print(f"Loading dataset: {DATASET} / {SPLIT}")
    ds = ViT_TTADataset(csv_path, cfg)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=vit_collate_fn, num_workers=4, pin_memory=True)
    
    print("Loading BLIP-2 Model...")
    ckpt = torch.load(DEFAULT_CHECKPOINT, map_location="cpu", weights_only=False)
    qformer   = QformerWrapper(device, is_eval=False, keep_vit=True).to(device)
    regressor = Regressor(input_dim=768, output_dim=1).to(device)
    
    qformer.model.Qformer.load_state_dict(ckpt["qformer.Qformer"], strict=True)
    qformer.model.query_tokens = nn.Parameter(ckpt["query_tokens"].to(device))
    regressor.load_state_dict(ckpt["regressor"], strict=True)
    
    # Save original ViT LN weights for reset
    ln_mods = [m for m in qformer.model.visual_encoder.modules() if isinstance(m, nn.LayerNorm)]
    ln_mods.extend([m for m in qformer.model.ln_vision.modules() if isinstance(m, nn.LayerNorm)])
    orig_ln = [{"w": m.weight.detach().clone(), "b": m.bias.detach().clone()} for m in ln_mods]
    
    print(f"\nRunning ViT TTA: lr={TTA_LR}, steps={TTA_STEPS}")
    
    all_rows = []
    
    for batch in tqdm(loader, desc="ViT TTA"):
        img_base = batch["img_base"].to(device)
        prompts = batch["prompts"]
        descs   = batch["descs"]
        gt_scores = batch["gt_scores"].numpy()
        names = batch["image_names"]
        B = img_base.size(0)
        
        # Reset model
        with torch.no_grad():
            for m, s in zip(ln_mods, orig_ln):
                m.weight.copy_(s["w"]); m.bias.copy_(s["b"])
        
        qformer.eval(); regressor.eval()
        with torch.no_grad():
            image_embeds = qformer.extract_image_embeds(img_base)
            mm = qformer.forward_qformer(image_embeds, prompts, descs)
            baseline_preds = regressor(mm).squeeze(-1).cpu().numpy()
        
        batch_rows = []
        for i in range(B):
            batch_rows.append({
                "image_name": names[i], "gt_score": gt_scores[i],
                "pred_baseline": baseline_preds[i],
            })
        
        # Unfreeze ONLY ViT LayerNorms
        for p in qformer.model.parameters(): p.requires_grad = False
        for p in regressor.parameters(): p.requires_grad = False
        
        vit_ln_params = []
        for m in qformer.model.visual_encoder.modules():
            if isinstance(m, nn.LayerNorm):
                m.weight.requires_grad = True
                m.bias.requires_grad = True
                vit_ln_params.extend([m.weight, m.bias])
                
        for m in qformer.model.ln_vision.modules():
            if isinstance(m, nn.LayerNorm):
                m.weight.requires_grad = True
                m.bias.requires_grad = True
                vit_ln_params.extend([m.weight, m.bias])
                
        optimizer = torch.optim.Adam(vit_ln_params, lr=TTA_LR)
        qformer.train()
        
        # TTA Steps
        for step in range(1, TTA_STEPS + 1):
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                # Forward ViT for base image
                embed_orig = qformer.model.ln_vision(qformer.model.visual_encoder(img_base.half())).float()
                cls_orig = embed_orig[:, 0, :]
                
                # GC Loss
                l_gc = gc_loss(cls_orig, torch.tensor(baseline_preds, device=device), p=0.25, temperature=0.5, device=device)
            
            if l_gc.item() > 0:
                l_gc.backward()
                optimizer.step()
                
            torch.cuda.empty_cache()
                
        # Final predictions using adapted ViT -> frozen Q-Former
        qformer.eval()
        with torch.no_grad():
            image_embeds_adapted = qformer.extract_image_embeds(img_base)
            mm_post = qformer.forward_qformer(image_embeds_adapted, prompts, descs)
            step_preds = regressor(mm_post).squeeze(-1).cpu().numpy()
        
        for i in range(B):
            batch_rows[i]["pred_tta"] = step_preds[i]
            
        all_rows.extend(batch_rows)
        torch.cuda.empty_cache()

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(OUT_DIR, "vit_tta_results.csv"), index=False)
    
    srcc_bl = spearmanr(df["pred_baseline"].values, df["gt_score"].values)
    plcc_bl = pearsonr(df["pred_baseline"].values, df["gt_score"].values)
    
    srcc_tta = spearmanr(df["pred_tta"].values, df["gt_score"].values)
    plcc_tta = pearsonr(df["pred_tta"].values, df["gt_score"].values)
    
    print("\n--- Final Results ---")
    print(f"Baseline : SRCC = {srcc_bl:.4f}, PLCC = {plcc_bl:.4f}")
    print(f"ViT TTA  : SRCC = {srcc_tta:.4f}, PLCC = {plcc_tta:.4f}")
    print("---------------------\n")

if __name__ == "__main__":
    main()
