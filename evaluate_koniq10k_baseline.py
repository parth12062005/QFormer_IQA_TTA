import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from lavis.models import load_model_and_preprocess

# --- Monkey-patch for lavis + transformers >= 4.40 ---
import transformers.modeling_utils
if not hasattr(transformers.modeling_utils, "apply_chunking_to_forward"):
    transformers.modeling_utils.apply_chunking_to_forward = lambda *args, **kwargs: None


##### ------------- ####
#####  DEFAULTS
##### ------------- ####
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT = os.path.join(_SCRIPT_DIR, "checkpoints", "evalmi_baseline_qf.pth")
DEFAULT_CSV   = "/media/parth/021f75bf-bae8-49ef-86a5-28ca19171835/parth/archive/koniq10k_distributions_sets.csv"
DEFAULT_IMG_ROOT   = "/media/parth/021f75bf-bae8-49ef-86a5-28ca19171835/parth/archive/512x384"

IMG_COL    = "image_name"
GT_COL     = "MOS"
SPLIT_COL  = "set"

# TTA hyperparameters
TTA_STEPS = 1
TTA_LR = 1e-4
TAU = 0.5


##### ------------- ####
#####  UTILS
##### ------------- ####
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def rankdata_numpy(a):
    a = np.asarray(a)
    sorter = np.argsort(a)
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(a))
    a_sorted = a[sorter]
    obs = np.concatenate(([True], a_sorted[1:] != a_sorted[:-1]))
    dense_rank = np.cumsum(obs)
    counts = np.bincount(dense_rank)
    cumulative = np.cumsum(counts)
    ranks = (cumulative[dense_rank] + cumulative[dense_rank - 1] + 1) / 2.0
    return ranks[inv]

def spearmanr_numpy(x, y):
    x, y = np.asarray(x), np.asarray(y)
    assert x.shape == y.shape
    rx, ry = rankdata_numpy(x), rankdata_numpy(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = np.sqrt(np.sum(rx**2) * np.sum(ry**2))
    return np.nan if denom == 0 else float(np.sum(rx * ry) / denom)

def pearsonr_numpy(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    assert x.shape == y.shape
    xm, ym = x - x.mean(), y - y.mean()
    denom = np.sqrt(np.sum(xm**2) * np.sum(ym**2))
    return np.nan if denom == 0 else float(np.sum(xm * ym) / denom)


##### ------------- ####
#####  DATASET
##### ------------- ####
class Koniq10kDataset(Dataset):
    def __init__(self, csv_path, img_root, split="test"):
        df = pd.read_csv(csv_path)
        if split != "all":
            df = df[df[SPLIT_COL] == split]
        self.df = df.reset_index(drop=True)
        self.img_root = img_root
        
        self.image_tf = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])
        
        self.vgg_tf = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.img_root, str(row[IMG_COL]))
        image = Image.open(image_path).convert("RGB")
        
        clip_image = self.image_tf(image)
        vgg_image = self.vgg_tf(image)

        prompt = "" # KonIQ-10k does not have prompts
        desc   = "" 
        gt     = torch.tensor(float(row[GT_COL]), dtype=torch.float32)

        return {
            "clip_image": clip_image,
            "vgg_image": vgg_image,
            "prompt": prompt,
            "description": desc,
            "image_name": str(row[IMG_COL]),
            "gt_score": gt,
        }

def collate_fn(batch):
    return {
        "clip_images": torch.stack([b["clip_image"] for b in batch], dim=0),
        "vgg_images": torch.stack([b["vgg_image"] for b in batch], dim=0),
        "prompts": [b["prompt"] for b in batch],
        "descs": [b["description"] for b in batch],
        "image_names": [b["image_name"] for b in batch],
        "gt_scores": torch.stack([b["gt_score"] for b in batch], dim=0),
    }


##### ------------- ####
#####  MODELS
##### ------------- ####
class Regressor(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.layer(x)

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

class QformerWrapper(nn.Module):
    def __init__(self, device, is_eval=True):
        super().__init__()
        model, _, _ = load_model_and_preprocess(
            name="blip2_feature_extractor",
            model_type="pretrain",
            is_eval=is_eval,
            device=device,
        )
        self.model = model.to(device)
        self.device = device

    def extract_image_embeds(self, images):
        images = images.to(self.device)
        with torch.no_grad():
            with self.model.maybe_autocast():
                image_embeds_frozen = self.model.ln_vision(self.model.visual_encoder(images))
        return image_embeds_frozen.float()

    def forward_qformer(self, image_embeds_frozen, prompts, descs):
        B = image_embeds_frozen.size(0)
        image_atts = torch.ones(
            image_embeds_frozen.size()[:-1], dtype=torch.long, device=self.device
        )
        query_tokens = self.model.query_tokens.expand(B, -1, -1)
        text_prompt = self.model.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=self.device)
        mm_attention_mask = torch.cat([query_atts, text_prompt.attention_mask], dim=1)

        mm_out = self.model.Qformer.bert(
            text_prompt.input_ids,
            query_embeds=query_tokens,
            attention_mask=mm_attention_mask,
            encoder_hidden_states=image_embeds_frozen,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        mm_query_embeds = mm_out.last_hidden_state[:, : query_tokens.size(1), :]
        mm_mean_embeds = mm_query_embeds.mean(dim=1)
        return mm_mean_embeds

    def forward(self, images, prompts, descs):
        image_embeds = self.extract_image_embeds(images)
        return self.forward_qformer(image_embeds, prompts, descs)


##### ------------- ####
#####  TTA HELPERS
##### ------------- ####
"""
TTA Auxiliary Losses based on:
"Test Time Adaptation for Blind Image Quality Assessment" (ICCV 2023) 
by Subhadeep Roy, Shankhanil Mitra, Soma Biswas, and Rajiv Soundararajan.

1. Group Contrastive (GC) Loss:
   Groups the test batch into "high quality" and "low quality" pseudo-labels 
   based on the model's initial predictions. It minimizes the feature distance 
   within the same group and maximizes the distance between different groups.

2. Rank Loss:
   Operates at the sample level by applying two varying severities of a distortion 
   (e.g., Gaussian Blur). It enforces that the feature distance between the heavily 
   distorted image and the original image should be strictly greater than the 
   distance between the slightly distorted image and the original image.
"""

def compute_gc_loss(feats, preds, p=0.25, temperature=0.1):
    B = feats.size(0)
    k = max(2, int(B * p))
    if B < 4:
        return torch.tensor(0.0, device=feats.device, requires_grad=True)
    
    idx = torch.argsort(preds)
    low_feats = F.normalize(feats[idx[:k]], dim=-1)
    high_feats = F.normalize(feats[idx[-k:]], dim=-1)
    
    loss = 0.0
    for i in range(k):
        # Anchor is high_feats[i]
        pos_sim = torch.exp(torch.matmul(high_feats[i], high_feats.t()) / temperature)
        neg_sim = torch.exp(torch.matmul(high_feats[i], low_feats.t()) / temperature)
        pos_sim_sum = pos_sim.sum() - pos_sim[i] # remove self-similarity
        loss += -torch.log(pos_sim_sum / (pos_sim_sum + neg_sim.sum() + 1e-8))
        
        # Anchor is low_feats[i]
        pos_sim_low = torch.exp(torch.matmul(low_feats[i], low_feats.t()) / temperature)
        neg_sim_low = torch.exp(torch.matmul(low_feats[i], high_feats.t()) / temperature)
        pos_sim_low_sum = pos_sim_low.sum() - pos_sim_low[i] # remove self-similarity
        loss += -torch.log(pos_sim_low_sum / (pos_sim_low_sum + neg_sim_low.sum() + 1e-8))
        
    return loss / (2 * k)

def compute_rank_loss(feat_orig, feat_weak, feat_strong):
    dist_weak = F.pairwise_distance(feat_weak, feat_orig, p=2)
    dist_strong = F.pairwise_distance(feat_strong, feat_orig, p=2)
    
    # We want dist_strong > dist_weak -> dist_strong - dist_weak > 0
    # Equivalent to BCEWithLogitsLoss with target=1
    target = torch.ones_like(dist_strong)
    loss = F.binary_cross_entropy_with_logits(dist_strong - dist_weak, target)
    return loss

def get_layernorm_params(model):
    ln_params = []
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            ln_params.extend([module.weight, module.bias])
    return [p for p in ln_params if p is not None]


##### ------------- ####
#####  MAIN
##### ------------- ####
def main():
    parser = argparse.ArgumentParser(description="Evaluate Q-Former on KonIQ-10k (No TTA vs TTA)")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--csv_path", type=str, default=DEFAULT_CSV)
    parser.add_argument("--img_root", type=str, default=DEFAULT_IMG_ROOT)
    parser.add_argument("--split", type=str, default="test", help="'test', 'validation', 'training', or 'all'")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading models...")
    qformer   = QformerWrapper(device=device, is_eval=True).to(device)
    regressor = Regressor(input_dim=768, output_dim=1).to(device)
    proj_head = ProjectionHead(input_dim=768, hidden_dim=256, output_dim=128).to(device)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    qformer.model.Qformer.load_state_dict(ckpt["qformer.Qformer"], strict=True)
    qformer.model.query_tokens = nn.Parameter(ckpt["query_tokens"].to(device))
    regressor.load_state_dict(ckpt["regressor"], strict=True)
    
    if "proj_head" in ckpt:
        proj_head.load_state_dict(ckpt["proj_head"])
    else:
        print("Warning: proj_head not found in checkpoint. Initialized from scratch.")

    print("Checkpoint loaded successfully.")

    test_dataset = Koniq10kDataset(csv_path=args.csv_path, img_root=args.img_root, split=args.split)
    test_loader  = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Total samples for split '{args.split}': {len(test_dataset)}")

    ORIGINAL_QUERY_TOKENS = qformer.model.query_tokens.detach().clone()
    qf_layernorms = [m for m in qformer.model.Qformer.modules() if isinstance(m, nn.LayerNorm)]
    ORIGINAL_LAYERNORM_STATES = [
        {"weight": m.weight.detach().clone() if m.weight is not None else None,
         "bias": m.bias.detach().clone() if m.bias is not None else None}
        for m in qf_layernorms
    ]

    def reset_model_state():
        with torch.no_grad():
            qformer.model.query_tokens.copy_(ORIGINAL_QUERY_TOKENS)
            for m, state in zip(qf_layernorms, ORIGINAL_LAYERNORM_STATES):
                if m.weight is not None:
                    m.weight.copy_(state["weight"])
                if m.bias is not None:
                    m.bias.copy_(state["bias"])

    def evaluate(use_tta=False):
        qformer.eval()
        regressor.eval()
        proj_head.eval()

        rows, pred_list, gt_list = [], [], []
        desc_tag = "Eval TTA" if use_tta else "Eval No TTA"
        
        for batch in tqdm(test_loader, desc=desc_tag):
            clip_images = batch["clip_images"]
            vgg_images  = batch["vgg_images"]
            prompts     = batch["prompts"]
            descs       = batch["descs"]

            reset_model_state()
            
            # Precompute frozen image embeddings
            image_embeds = qformer.extract_image_embeds(clip_images)

            if use_tta and clip_images.size(0) > 1:
                # Initial prediction to get pseudo-labels for GC Loss
                with torch.no_grad():
                    init_embeds = qformer.forward_qformer(image_embeds, prompts, descs)
                    init_preds = regressor(init_embeds).squeeze(-1)
                
                # Create augmented images for Rank Loss (Weak & Strong Blur)
                with torch.no_grad():
                    blur_weak = transforms.GaussianBlur(kernel_size=5, sigma=(5.0, 10.0))
                    blur_strong = transforms.GaussianBlur(kernel_size=5, sigma=(30.0, 40.0))
                    
                    images_weak = blur_weak(clip_images)
                    images_strong = blur_strong(clip_images)
                    
                    embeds_weak = qformer.extract_image_embeds(images_weak)
                    embeds_strong = qformer.extract_image_embeds(images_strong)

                # Setup optimizer for query_tokens and layernorms ONLY
                params_to_update = [qformer.model.query_tokens] + get_layernorm_params(qformer.model.Qformer)
                
                # Ensure grad is enabled for these params
                for p in params_to_update:
                    p.requires_grad = True

                # Everything else is frozen
                params_to_update_ids = {id(p) for p in params_to_update}
                for name, p in qformer.named_parameters():
                    if id(p) not in params_to_update_ids:
                        p.requires_grad = False
                
                optimizer = optim.Adam(params_to_update, lr=TTA_LR)
                
                qformer.train()
                proj_head.train()

                for step in range(TTA_STEPS):
                    optimizer.zero_grad()
                    
                    # 1. Forward passes for original and augmented images
                    mm_embeds_orig = qformer.forward_qformer(image_embeds, prompts, descs)
                    mm_embeds_weak = qformer.forward_qformer(embeds_weak, prompts, descs)
                    mm_embeds_strong = qformer.forward_qformer(embeds_strong, prompts, descs)
                    
                    proj_feats_orig = proj_head(mm_embeds_orig)
                    proj_feats_weak = proj_head(mm_embeds_weak)
                    proj_feats_strong = proj_head(mm_embeds_strong)
                    
                    # 2. Compute Rank Loss
                    rank_loss = compute_rank_loss(proj_feats_orig, proj_feats_weak, proj_feats_strong)
                    
                    # 3. Compute Group Contrastive (GC) Loss
                    gc_loss = compute_gc_loss(proj_feats_orig, init_preds, p=0.25, temperature=TAU)
                    
                    total_loss = rank_loss + gc_loss
                    
                    if total_loss > 0:
                        total_loss.backward()
                        optimizer.step()

                qformer.eval()
                proj_head.eval()

            # Final Prediction
            with torch.no_grad():
                mm_embeds = qformer.forward_qformer(image_embeds, prompts, descs)
                pred = regressor(mm_embeds).squeeze(-1)

            pred_list.append(pred.cpu())
            gt_list.append(batch["gt_scores"])

            pred_cpu = pred.cpu().numpy()
            gt_cpu = batch["gt_scores"].numpy()
            for i in range(len(batch["image_names"])):
                rows.append({
                    "image_name": batch["image_names"][i],
                    "gt_score": float(gt_cpu[i]),
                    "pred_score": float(pred_cpu[i]),
                })

        preds = torch.cat(pred_list).numpy()
        gts   = torch.cat(gt_list).numpy()
        srcc  = spearmanr_numpy(preds, gts)
        plcc  = pearsonr_numpy(preds, gts)
        df = pd.DataFrame(rows)
        return srcc, plcc, df

    # Run without TTA
    print(f"\n--- Evaluation on KonIQ-10k ({args.split} split) (Without TTA) ---")
    srcc_notta, plcc_notta, df_notta = evaluate(use_tta=False)
    
    # Run with TTA
    print(f"\n--- Evaluation on KonIQ-10k ({args.split} split) (With TTA) ---")
    print("\n[TTA INFO] Unfrozen Parameters for TTA:")
    unfrozen_total = 0
    q_tokens = qformer.model.query_tokens
    print(f"  - query_tokens: {list(q_tokens.shape)} -> {q_tokens.numel()} params")
    unfrozen_total += q_tokens.numel()
    
    ln_count = 0
    ln_params = 0
    for name, module in qformer.model.Qformer.named_modules():
        if isinstance(module, nn.LayerNorm):
            ln_count += 1
            if module.weight is not None:
                ln_params += module.weight.numel()
            if module.bias is not None:
                ln_params += module.bias.numel()
    print(f"  - Q-Former LayerNorms ({ln_count} layers): {ln_params} params")
    unfrozen_total += ln_params
    
    print(f"[TTA INFO] Total params updated per step: {unfrozen_total}\n")
    srcc_tta, plcc_tta, df_tta = evaluate(use_tta=True)

    # Save to CSV
    csv_notta = os.path.join(_SCRIPT_DIR, f"koniq10k_{args.split}_no_tta_results.csv")
    csv_tta   = os.path.join(_SCRIPT_DIR, f"koniq10k_{args.split}_with_tta_results.csv")
    df_notta.to_csv(csv_notta, index=False)
    df_tta.to_csv(csv_tta, index=False)

    print("\n" + "=" * 50)
    print(f"  EVALUATION RESULTS (KonIQ-10k - {args.split})")
    print("=" * 50)
    print(f"  [No TTA]   SRCC : {srcc_notta:.6f} | PLCC : {plcc_notta:.6f}")
    print(f"  [With TTA] SRCC : {srcc_tta:.6f} | PLCC : {plcc_tta:.6f}")
    print("=" * 50)
    print(f"Saved NO TTA results to: {csv_notta}")
    print(f"Saved WITH TTA results to: {csv_tta}")

if __name__ == "__main__":
    set_seed(1234)
    main()
