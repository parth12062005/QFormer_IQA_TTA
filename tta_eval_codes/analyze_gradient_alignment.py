import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
from skimage.util import random_noise
from tqdm import tqdm

from lavis.models import load_model_and_preprocess

# Paths
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
IMG_ROOT = "/media/parth/021f75bf-bae8-49ef-86a5-28ca19171835/parth/agiqa-20k/AIGCQA-30K-Image"
SPLIT_CSV = os.path.join(_PROJECT_DIR, "important split files-20260527T062853Z-3-001", "important split files", "A20K_new", "A20k_train_full_PT1_normalized.csv")
CHECKPOINT = os.path.join(_PROJECT_DIR, "checkpoints", "evalmi_baseline_qf.pth")
GRADIENT_FILE = os.path.join(_SCRIPT_DIR, "results", "finetune_layernorm_gradient_tracker", "layernorm_gradients.pt")

def _build_image_lookup(img_root):
    lookup = {}
    for subdir in os.listdir(img_root):
        subdir_path = os.path.join(img_root, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for fname in os.listdir(subdir_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                lookup[fname] = os.path.join(subdir_path, fname)
    return lookup

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def augment_image(image_path, aug_type):
    if aug_type == "orig":
        return Image.open(image_path).convert("RGB")
    elif "blur" in aug_type:
        img = Image.open(image_path).convert("RGB")
        sigma = 1 + np.random.random() * 19 if aug_type == "blur_low" else 40 + np.random.random() * 40
        return TF.gaussian_blur(img, kernel_size=[5, 5], sigma=[sigma, sigma])
    elif "comp" in aug_type:
        img = Image.open(image_path).convert("RGB")
        q = int(80 + np.random.random() * 10) if aug_type == "comp_low" else int(40 + np.random.random() * 20)
        import io
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    elif "nos" in aug_type:
        ab = cv2.imread(image_path)
        ab = cv2.cvtColor(ab, cv2.COLOR_BGR2RGB)
        sigma = 0.00001 + np.random.random() * 0.000001 if aug_type == "nos_low" else 0.00005 + np.random.random() * 0.000001
        noise = random_noise(ab, mode='gaussian', var=sigma)
        return Image.fromarray((noise * 255).astype('uint8'))

class A20KRawDataset(Dataset):
    def __init__(self, df, image_lookup, vis_processors):
        self.df = df
        self.image_lookup = image_lookup
        self.vis_processors = vis_processors

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row['image_name'])
        path = self.image_lookup[img_name]
        
        # Original
        orig_img = augment_image(path, "orig")
        orig_t = self.vis_processors["eval"](orig_img)
        
        # Rank Augmentations
        aug_types = ["comp_low", "comp_high", "nos_low", "nos_high", "blur_low", "blur_high"]
        augs_t = []
        for at in aug_types:
            augs_t.append(self.vis_processors["eval"](augment_image(path, at)))

        return {
            "orig": orig_t,
            "augs": augs_t,
            "prompt": str(row['prompt']),
            "desc": str(row['gen_answer']),
            "gt_score": torch.tensor(float(row['gt_score']), dtype=torch.float32)
        }

def collate_fn(batch):
    return {
        "orig": torch.stack([b["orig"] for b in batch]),
        "augs": [torch.stack([b["augs"][i] for b in batch]) for i in range(6)],
        "prompts": [b["prompt"] for b in batch],
        "descs": [b["desc"] for b in batch],
        "gt_scores": torch.stack([b["gt_score"] for b in batch])
    }

class RegressorLinear(nn.Module):
    def __init__(self, input_dim=768, output_dim=1):
        super().__init__()
        self.layer = nn.Linear(input_dim, output_dim)
    def forward(self, x): return self.layer(x)

def detect_regressor_type(ckpt):
    return "linear" if "layer.weight" in ckpt["regressor"].keys() else "mlp"

# Losses
class GroupContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool)).float())
        self.register_buffer("positives_mask", (~torch.eye(batch_size * 1, batch_size * 1, dtype=torch.bool)).float())

    def forward(self, emb_i, emb_j):
        self.negatives_mask[:len(emb_i), :len(emb_j)] = False
        self.negatives_mask[len(emb_i):, len(emb_j):] = False

        z_i = torch.nn.functional.normalize(emb_i, dim=1)
        z_j = torch.nn.functional.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = torch.nn.functional.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        pos_similarity_matrix = similarity_matrix[:len(emb_i), :len(emb_j)] * self.positives_mask
        neg_similarity_matrix = similarity_matrix[len(emb_i):, len(emb_j):] * self.positives_mask

        sim_ij = torch.sum(pos_similarity_matrix, dim=1) / max(1, len(neg_similarity_matrix) - 1)
        sim_ji = torch.sum(neg_similarity_matrix, dim=1) / max(1, len(neg_similarity_matrix) - 1)

        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = torch.sum(nominator / (nominator + torch.sum(denominator, dim=1))) / (2 * self.batch_size)
        return -torch.log(loss_partial)

def main():
    print("Loading image paths...")
    image_lookup = _build_image_lookup(IMG_ROOT)
    
    print("Loading dataset...")
    df = pd.read_csv(SPLIT_CSV)
    df.columns = df.columns.str.strip()
    
    # Exact reproduction of finetune_and_track_gradients.py dataset
    set_seed(1234)
    train_df_sampled = df.sample(frac=0.20, random_state=1234).reset_index(drop=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading model...")
    model, vis_processors, _ = load_model_and_preprocess("blip2_feature_extractor", "pretrain", is_eval=False, device=device)
    model = model.float()
    
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    regressor = RegressorLinear(768, 1).to(device)
    
    model.Qformer.load_state_dict(ckpt["qformer.Qformer"], strict=True)
    model.query_tokens = nn.Parameter(ckpt["query_tokens"].to(device))
    regressor.load_state_dict(ckpt["regressor"], strict=True)
    
    # Only unfreeze LayerNorms (like the tracker run)
    for p in model.parameters(): p.requires_grad = False
    for p in regressor.parameters(): p.requires_grad = False
    
    model.query_tokens.requires_grad = True
    for p in regressor.parameters(): p.requires_grad = True
    for m in model.Qformer.modules():
        if isinstance(m, nn.LayerNorm):
            for p in m.parameters(): p.requires_grad = True

    dataset = A20KRawDataset(train_df_sampled, image_lookup, vis_processors)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=4)
    
    print("Loading saved Oracle gradients...")
    oracle_grads_list = torch.load(GRADIENT_FILE, map_location="cpu")
    
    rank_loss_fn = nn.BCELoss()
    m_sigmoid = nn.Sigmoid()
    mse_loss = nn.MSELoss()
    
    results = []
    
    model.train()
    regressor.train()
    
    def qformer_forward(images, prompts, descs):
        B = images.size(0)
        feats = []
        chunk_size = 4
        for i in range(0, B, chunk_size):
            sub_images = images[i:i+chunk_size]
            sub_prompts = prompts[i:i+chunk_size]
            sub_descs = descs[i:i+chunk_size]
            
            image_embeds = model.ln_vision(model.visual_encoder(sub_images))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)
            sub_query_tokens = model.query_tokens.expand(sub_images.size(0), -1, -1)
            text_prompt = model.tokenizer(sub_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            query_atts = torch.ones(sub_query_tokens.size()[:-1], dtype=torch.long, device=device)
            mm_attention_mask = torch.cat([query_atts, text_prompt.attention_mask], dim=1)
            
            mm_out = model.Qformer.bert(
                text_prompt.input_ids, query_embeds=sub_query_tokens, attention_mask=mm_attention_mask,
                encoder_hidden_states=image_embeds, encoder_attention_mask=image_atts, return_dict=True
            )
            feats.append(mm_out.last_hidden_state[:, :sub_query_tokens.size(1), :].mean(dim=1))
        return torch.cat(feats, dim=0)

    print("Starting analysis on first 30 batches...")
    for step, batch in enumerate(dataloader):
        if step >= 30: break
        
        orig_imgs = batch["orig"].to(device)
        prompts = batch["prompts"]
        descs = batch["descs"]
        gts = batch["gt_scores"].to(device)
        
        model.zero_grad()
        regressor.zero_grad()
        
        # 1. Oracle Gradient (MSE)
        feat_orig_oracle = qformer_forward(orig_imgs, prompts, descs)
        pred_orig_oracle = regressor(feat_orig_oracle).squeeze(-1)
        loss_oracle = mse_loss(pred_orig_oracle, gts)
        loss_oracle.backward()
        
        grad_oracle = []
        for name, p in model.Qformer.named_parameters():
            if "LayerNorm" in name and p.requires_grad and p.grad is not None:
                grad_oracle.append(p.grad.detach().cpu().flatten())
        grad_oracle_vec = torch.cat(grad_oracle) if grad_oracle else torch.tensor([0.0])
        
        # Verify it matches the saved gradient
        saved_grad_dict = oracle_grads_list[step]
        saved_grad_vec = []
        for name, _ in model.Qformer.named_parameters():
            if "LayerNorm" in name and name in saved_grad_dict:
                saved_grad_vec.append(saved_grad_dict[name].flatten())
        saved_grad_vec = torch.cat(saved_grad_vec) if saved_grad_vec else torch.tensor([0.0])
        
        # Compute difference to ensure our reconstruction is perfect
        diff = torch.norm(grad_oracle_vec - saved_grad_vec).item()
        
        model.zero_grad()
        regressor.zero_grad()
        
        # 2. Proxy TTA Gradient (GC + Rank)
        # We need preds to sort for GC
        with torch.no_grad():
            feat_orig_no_grad = qformer_forward(orig_imgs, prompts, descs)
            pred0 = regressor(feat_orig_no_grad).squeeze(-1)
            
        # GC Loss
        if orig_imgs.size(0) >= 4:
            feat_orig_gc = qformer_forward(orig_imgs, prompts, descs)
            idx = torch.argsort(pred0)
            f_pos, f_neg = [], []
            for n in range(orig_imgs.size(0) // 4):
                f_pos.append(feat_orig_gc[idx[n]])
                f_neg.append(feat_orig_gc[idx[-n-1]])
            if f_pos:
                f_pos = torch.stack(f_pos)
                f_neg = torch.stack(f_neg)
                loss_fn_gc = GroupContrastiveLoss(f_pos.size(0), 1).to(device)
                loss_gc = loss_fn_gc(f_neg, f_pos)
                loss_gc.backward()
                
        # Rank Loss
        target_rank = torch.ones(orig_imgs.size(0)).to(device)
        for i in range(0, 6, 2): # comp, nos, blur
            feat_orig_rank = qformer_forward(orig_imgs, prompts, descs)
            feat_low = qformer_forward(batch["augs"][i].to(device), prompts, descs)
            feat_high = qformer_forward(batch["augs"][i+1].to(device), prompts, descs)
            
            dist_high = torch.nn.PairwiseDistance(p=2)(feat_high, feat_orig_rank)
            dist_low = torch.nn.PairwiseDistance(p=2)(feat_low, feat_orig_rank)
            loss_rank = rank_loss_fn(m_sigmoid(dist_high - dist_low), target_rank)
            loss_rank.backward()
            
        grad_tta = []
        for name, p in model.Qformer.named_parameters():
            if "LayerNorm" in name and p.requires_grad and p.grad is not None:
                grad_tta.append(p.grad.detach().cpu().flatten())
        grad_tta_vec = torch.cat(grad_tta) if grad_tta else torch.tensor([0.0])
        
        cos_sim = torch.nn.functional.cosine_similarity(saved_grad_vec.unsqueeze(0), grad_tta_vec.unsqueeze(0)).item()
            
        results.append(cos_sim)
        print(f"Batch {step+1}: Oracle Match Diff = {diff:.6f}, TTA Cosine Sim = {cos_sim:.4f}")

    print(f"\nAverage Cosine Similarity (first 30 batches): {np.mean(results):.4f}")

if __name__ == "__main__":
    main()
