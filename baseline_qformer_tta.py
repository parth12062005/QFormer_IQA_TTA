##### TTA Script: Feature Affinity-based Group Contrastive (FAGC) Loss
##
#### Test-Time Adaptation for baseline Q-Former IQA
#### Only query_tokens (32x768) + projection head are updated via backprop
#### Based on: "Feature Affinity based Clustering for TTA for IQA" (ICME 2025)
####
#### Pipeline (episodic, per-batch):
####   1. Save original query_tokens
####   2. For each test batch:
####      a. Cluster using VGG-16 feature affinity (not base model scores)
####      b. Compute FAGC loss in projection space
####      c. Backprop to update query_tokens + projection head
####      d. Predict quality scores with adapted model
####      e. Reset query_tokens for next batch

##### ----------------- ####
#####  1) IMPORTS + UTILS
##### ----------------- ####
import os
import copy
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from lavis.models import load_model_and_preprocess

##### SET SEED #####
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 1234
set_seed(SEED)

#### METRICS ####
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
    return np.nan if denom == 0 else np.sum(rx * ry) / denom


##### ----------------- ####
#####  2) DATASET + COLLATE
##### ----------------- ####
IMG_COL    = "image_name"
PROMPT_COL = "prompt"
DESC_COL   = "gen_answer"
GT_COL     = "gt_score"

class QFormerQualityDataset(Dataset):
    def __init__(self, img_root, csv_path=None, df=None, image_tf=None):
        assert (csv_path is not None) ^ (df is not None), "Provide exactly one of csv_path or df"
        self.df = pd.read_csv(csv_path) if csv_path is not None else df.reset_index(drop=True)
        self.df.columns = self.df.columns.str.strip()
        self.img_root = img_root

        if image_tf is None:
            self.image_tf = transforms.Compose([
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ])
        else:
            self.image_tf = image_tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.img_root, str(row[IMG_COL]))
        image = Image.open(image_path).convert("RGB")
        image = self.image_tf(image)
        prompt = str(row[PROMPT_COL])
        desc   = str(row[DESC_COL])
        gt     = torch.tensor(float(row[GT_COL]), dtype=torch.float32)
        return {
            "image": image, "prompt": prompt, "description": desc,
            "image_name": str(row[IMG_COL]), "gt_score": gt,
        }

def collate_fn(batch):
    return {
        "images": torch.stack([b["image"] for b in batch], dim=0),
        "prompts": [b["prompt"] for b in batch],
        "descs": [b["description"] for b in batch],
        "image_names": [b["image_name"] for b in batch],
        "gt_scores": torch.stack([b["gt_score"] for b in batch], dim=0),
    }


##### ----------------- ####
#####  3) DEVICE
##### ----------------- ####
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


##### ----------------- ####
#####  4) MODELS
##### ----------------- ####
class Regressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x):
        return self.layer(x)


class ProjectionHead(nn.Module):
    """Small MLP that projects mm_mean_embeds into a lower-dim space for FAGC loss."""
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)  # L2-normalize for cosine sim


class QformerWrapper(nn.Module):
    def __init__(self, device, is_eval=False):
        super().__init__()
        model, vis_proc, txt_proc = load_model_and_preprocess(
            name="blip2_feature_extractor",
            model_type="pretrain",
            is_eval=is_eval,
            device=device,
        )
        self.model = model.to(device)
        self.device = device

    def forward(self, images, prompts, descs):
        B = images.size(0)
        images = images.to(self.device)

        with torch.no_grad():
            with self.model.maybe_autocast():
                image_embeds_frozen = self.model.ln_vision(self.model.visual_encoder(images))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long, device=self.device
            )

        query_tokens = self.model.query_tokens.expand(B, -1, -1)  # [B,32,768]

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

        mm_query_embeds = mm_out.last_hidden_state[:, : query_tokens.size(1), :]  # [B,32,768]
        mm_mean_embeds = mm_query_embeds.mean(dim=1)                               # [B,768]
        return mm_mean_embeds


##### ----------------- ####
#####  5) VGG-16 FEATURE EXTRACTOR (frozen, for clustering only)
##### ----------------- ####
class VGGFeatureExtractor(nn.Module):
    """Frozen VGG-16 conv features → adaptive avg pool → flat vector."""
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg16(pretrained=True)
        self.features = vgg.features  # conv layers only
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.to(device)
        self.eval()
        for p in self.parameters():
            p.requires_grad = False
        self._device = device

        # VGG expects ImageNet normalization; input images are already CLIP-normalized,
        # so we re-normalize: undo CLIP norm → apply ImageNet norm
        clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1,3,1,1)
        clip_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1,3,1,1)
        inet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
        inet_std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
        self.register_buffer("clip_mean", clip_mean)
        self.register_buffer("clip_std", clip_std)
        self.register_buffer("inet_mean", inet_mean)
        self.register_buffer("inet_std", inet_std)

    @torch.no_grad()
    def forward(self, x):
        # x: [B,3,224,224] CLIP-normalized
        x = x * self.clip_std + self.clip_mean          # undo CLIP norm → [0,1]
        x = (x - self.inet_mean) / self.inet_std        # apply ImageNet norm
        x = self.features(x)                             # [B, 512, 7, 7]
        x = self.pool(x).flatten(1)                      # [B, 512]
        return F.normalize(x, dim=-1)                    # L2-norm for cosine sim


##### ----------------- ####
#####  6) FAGC LOSS
##### ----------------- ####
def compute_fagc_loss(proj_feats, cluster_c1_idx, cluster_c2_idx, temperature=0.5):
    """
    Feature Affinity-based Group Contrastive loss (Eq. 2 from the paper).

    For each sample i in C1:
        L = -log( exp(sim(l_i, l_j)/τ) / Σ_{k∈C2} exp(sim(l_i, l_k)/τ) )
        where j is another sample in C1 (positive pair).

    Symmetrically computed for C2 as well.

    Args:
        proj_feats: [B, D] L2-normalized projection features
        cluster_c1_idx: list of indices in high-quality cluster
        cluster_c2_idx: list of indices in low-quality cluster
        temperature: τ
    Returns:
        scalar loss
    """
    loss = torch.tensor(0.0, device=proj_feats.device)
    count = 0

    # -- Loss for C1 samples (positives from C1, negatives from C2) --
    if len(cluster_c1_idx) >= 2 and len(cluster_c2_idx) >= 1:
        for i in cluster_c1_idx:
            # positive: mean similarity with other C1 members
            pos_indices = [j for j in cluster_c1_idx if j != i]
            if len(pos_indices) == 0:
                continue
            pos_sims = torch.stack([
                F.cosine_similarity(proj_feats[i].unsqueeze(0), proj_feats[j].unsqueeze(0))
                for j in pos_indices
            ])  # [num_pos]
            # use mean of positive sims as numerator (average over all positives)
            pos_logit = (pos_sims / temperature).logsumexp(0) - np.log(len(pos_indices))

            neg_sims = torch.stack([
                F.cosine_similarity(proj_feats[i].unsqueeze(0), proj_feats[k].unsqueeze(0))
                for k in cluster_c2_idx
            ])  # [num_neg]
            neg_logits = neg_sims / temperature  # [num_neg]

            # log-sum-exp over negatives
            denom = torch.cat([pos_logit.unsqueeze(0), neg_logits.squeeze()])
            loss += -pos_logit + torch.logsumexp(denom, dim=0)
            count += 1

    # -- Loss for C2 samples (positives from C2, negatives from C1) --
    if len(cluster_c2_idx) >= 2 and len(cluster_c1_idx) >= 1:
        for i in cluster_c2_idx:
            pos_indices = [j for j in cluster_c2_idx if j != i]
            if len(pos_indices) == 0:
                continue
            pos_sims = torch.stack([
                F.cosine_similarity(proj_feats[i].unsqueeze(0), proj_feats[j].unsqueeze(0))
                for j in pos_indices
            ])
            pos_logit = (pos_sims / temperature).logsumexp(0) - np.log(len(pos_indices))

            neg_sims = torch.stack([
                F.cosine_similarity(proj_feats[i].unsqueeze(0), proj_feats[k].unsqueeze(0))
                for k in cluster_c1_idx
            ])
            neg_logits = neg_sims / temperature

            denom = torch.cat([pos_logit.unsqueeze(0), neg_logits.squeeze()])
            loss += -pos_logit + torch.logsumexp(denom, dim=0)
            count += 1

    return loss / max(count, 1)


def cluster_by_vgg_affinity(vgg_feats, anchor_idx):
    """
    Cluster images by cosine affinity to the anchor (highest predicted quality).
    Top half → C1 (high quality), bottom half → C2 (low quality).
    Anchor is always in C1.

    Args:
        vgg_feats: [B, 512] L2-normalized VGG features
        anchor_idx: int, index of the highest quality image
    Returns:
        c1_idx, c2_idx: lists of indices
    """
    B = vgg_feats.size(0)
    if B < 2:
        return list(range(B)), []

    anchor_feat = vgg_feats[anchor_idx].unsqueeze(0)  # [1, 512]
    affinities = F.cosine_similarity(anchor_feat, vgg_feats, dim=-1)  # [B]

    # Sort by affinity descending; top half is C1
    sorted_indices = torch.argsort(affinities, descending=True).cpu().tolist()
    half = B // 2
    c1_idx = sorted_indices[:half]
    c2_idx = sorted_indices[half:]

    # Ensure anchor is in C1
    if anchor_idx in c2_idx:
        c2_idx.remove(anchor_idx)
        c1_idx.append(anchor_idx)

    return c1_idx, c2_idx


##### ----------------- ####
#####  7) INIT MODELS + LOAD CHECKPOINT
##### ----------------- ####
qformer = QformerWrapper(device=device, is_eval=False).to(device)
regressor = Regressor(input_dim=768, hidden_dim=256, output_dim=1).to(device)
proj_head = ProjectionHead(input_dim=768, hidden_dim=256, output_dim=128).to(device)
vgg_extractor = VGGFeatureExtractor(device=device)

def load_checkpoint(checkpoint_path: str):
    """Load pretrained baseline Q-Former checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    for key in ["qformer.Qformer", "query_tokens", "regressor"]:
        if key not in ckpt:
            raise KeyError(f"Missing key '{key}' in checkpoint.")

    qformer.model.Qformer.load_state_dict(ckpt["qformer.Qformer"], strict=True)

    qt = ckpt["query_tokens"]
    qt = qt.to(qformer.model.query_tokens.device, dtype=qformer.model.query_tokens.dtype)
    with torch.no_grad():
        qformer.model.query_tokens.copy_(qt)

    regressor.load_state_dict(ckpt["regressor"], strict=True)
    print(f"[Loaded] Qformer+query_tokens+regressor from: {checkpoint_path}")


# ---- Load pretrained checkpoint ----
INIT_CKPT = "/home/rajivs/anatapmitra/anatap_data/Qformer_experiments/new_pretraining/evalmi_baseline_qf_ver2.pth"
load_checkpoint(INIT_CKPT)

# ---- Freeze EVERYTHING ----
for p in qformer.model.parameters():
    p.requires_grad = False
for p in regressor.parameters():
    p.requires_grad = False

# ---- Unfreeze ONLY query_tokens ----
qformer.model.query_tokens.requires_grad = True

# ---- Projection head is trainable ----
for p in proj_head.parameters():
    p.requires_grad = True

# Save original query_tokens for episodic reset
ORIGINAL_QUERY_TOKENS = qformer.model.query_tokens.detach().clone()


##### ----------------- ####
#####  8) TTA STEP
##### ----------------- ####
TTA_STEPS = 1          # number of adaptation steps per batch
TTA_LR    = 1e-4
TAU       = 0.5        # temperature for FAGC loss

def tta_adapt_batch(images, prompts, descs):
    """
    Perform episodic TTA on a single batch:
      1. Reset query_tokens to pretrained state
      2. Reset projection head
      3. Run TTA_STEPS of FAGC adaptation
      4. Return adapted mm_mean_embeds for prediction
    """
    # ---- Episodic reset ----
    with torch.no_grad():
        qformer.model.query_tokens.copy_(ORIGINAL_QUERY_TOKENS)
    # Re-init projection head each batch (fresh head per episode)
    for layer in proj_head.net:
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

    # ---- Setup optimizer for this episode ----
    tta_optimizer = torch.optim.Adam(
        [qformer.model.query_tokens] + list(proj_head.parameters()),
        lr=TTA_LR,
    )

    images = images.to(device)
    B = images.size(0)

    # ---- VGG features for clustering (computed once, frozen) ----
    vgg_feats = vgg_extractor(images)  # [B, 512], L2-normed, no grad

    # ---- Get initial predicted scores to find anchor ----
    with torch.no_grad():
        qformer.eval()
        regressor.eval()
        init_embeds = qformer(images, prompts, descs)
        init_scores = regressor(init_embeds).squeeze(-1)  # [B]
        anchor_idx = int(torch.argmax(init_scores).item())

    # ---- Cluster by VGG affinity ----
    c1_idx, c2_idx = cluster_by_vgg_affinity(vgg_feats, anchor_idx)

    # Skip TTA if we can't form two non-empty clusters
    if len(c1_idx) < 2 or len(c2_idx) < 1:
        with torch.no_grad():
            qformer.eval()
            return qformer(images, prompts, descs)

    # ---- TTA adaptation steps ----
    qformer.train()
    proj_head.train()

    for step in range(TTA_STEPS):
        tta_optimizer.zero_grad()

        mm_mean_embeds = qformer(images, prompts, descs)  # [B, 768]
        proj_feats = proj_head(mm_mean_embeds)              # [B, 128], L2-normed

        fagc_loss = compute_fagc_loss(proj_feats, c1_idx, c2_idx, temperature=TAU)
        fagc_loss.backward()
        tta_optimizer.step()

    # ---- Final forward pass with adapted query_tokens ----
    qformer.eval()
    with torch.no_grad():
        adapted_embeds = qformer(images, prompts, descs)
    return adapted_embeds


##### ----------------- ####
#####  9) EVALUATE WITH TTA
##### ----------------- ####
@torch.no_grad()
def evaluate_no_tta(dataloader, output_csv_path=None, desc_tag="Eval (no TTA)"):
    """Baseline evaluation without TTA for comparison."""
    qformer.eval()
    regressor.eval()
    with torch.no_grad():
        qformer.model.query_tokens.copy_(ORIGINAL_QUERY_TOKENS)

    rows, pred_list, gt_list = [], [], []
    for batch in tqdm(dataloader, total=len(dataloader), desc=desc_tag):
        images = batch["images"].to(device)
        mm_embeds = qformer(images, batch["prompts"], batch["descs"])
        pred = regressor(mm_embeds).squeeze(-1)

        pred_list.append(pred.cpu())
        gt_list.append(batch["gt_scores"])
        pred_cpu = pred.cpu().numpy()
        gt_cpu = batch["gt_scores"].numpy()
        for i in range(len(batch["image_names"])):
            rows.append({
                "image_name": batch["image_names"][i],
                "prompt": batch["prompts"][i],
                "gen_answer": batch["descs"][i],
                "gt_score": float(gt_cpu[i]),
                "pred_score": float(pred_cpu[i]),
            })

    preds = torch.cat(pred_list).numpy()
    gts   = torch.cat(gt_list).numpy()
    srcc  = spearmanr_numpy(preds, gts)

    df = pd.DataFrame(rows)
    if output_csv_path:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        print(f"[Saved] {output_csv_path}")
    return srcc, df


def evaluate_with_tta(dataloader, output_csv_path=None, desc_tag="Eval (FAGC TTA)"):
    """Episodic TTA evaluation: adapt per batch, predict, reset."""
    regressor.eval()

    rows, pred_list, gt_list = [], [], []
    for batch in tqdm(dataloader, total=len(dataloader), desc=desc_tag):
        images = batch["images"]
        prompts = batch["prompts"]
        descs = batch["descs"]

        # Adapt and get embeddings
        adapted_embeds = tta_adapt_batch(images, prompts, descs)

        with torch.no_grad():
            pred = regressor(adapted_embeds).squeeze(-1)

        pred_list.append(pred.cpu())
        gt_list.append(batch["gt_scores"])
        pred_cpu = pred.cpu().numpy()
        gt_cpu = batch["gt_scores"].numpy()
        for i in range(len(batch["image_names"])):
            rows.append({
                "image_name": batch["image_names"][i],
                "prompt": batch["prompts"][i],
                "gen_answer": batch["descs"][i],
                "gt_score": float(gt_cpu[i]),
                "pred_score": float(pred_cpu[i]),
            })

    preds = torch.cat(pred_list).numpy()
    gts   = torch.cat(gt_list).numpy()
    srcc  = spearmanr_numpy(preds, gts)

    df = pd.DataFrame(rows)
    if output_csv_path:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        print(f"[Saved] {output_csv_path}")
    return srcc, df


##### ----------------- ####
#####  10) MAIN
##### ----------------- ####
if __name__ == "__main__":
    print("=" * 60)
    print("FAGC TTA FOR BASELINE QFORMER")
    print("Adapting: query_tokens only | Loss: FAGC | Episodic reset")
    print("=" * 60)

    # ----- Configure test dataset here -----
    test_csv  = "/home/rajivs/anatapmitra/anatap_data/generated_descriptions_PT1/A20K_new/A20k_test_full_PT1_normalized.csv"
    img_root  = "/home/rajivs/anatapmitra/anatap_data/anatap_1/aigiqa-20k/Images/train"
    out_csv_no_tta  = "/home/rajivs/anatapmitra/anatap_data/Qformer_experiments/TTA/a20k_baseline_qf_no_tta.csv"
    out_csv_tta     = "/home/rajivs/anatapmitra/anatap_data/Qformer_experiments/TTA/a20k_baseline_qf_fagc_tta.csv"

    test_dataset = QFormerQualityDataset(csv_path=test_csv, img_root=img_root)
    # Batch size = 8 as in the paper
    test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    print(f"Test set size: {len(test_dataset)}")

    # ---- Evaluate WITHOUT TTA (baseline) ----
    print("\n--- Evaluation WITHOUT TTA ---")
    srcc_no_tta, _ = evaluate_no_tta(test_loader, output_csv_path=out_csv_no_tta)
    print(f"SRCC (no TTA): {srcc_no_tta:.5f}")

    # ---- Evaluate WITH FAGC TTA ----
    print("\n--- Evaluation WITH FAGC TTA ---")
    srcc_tta, _ = evaluate_with_tta(test_loader, output_csv_path=out_csv_tta)
    print(f"SRCC (FAGC TTA): {srcc_tta:.5f}")

    print("\n" + "=" * 60)
    print(f"SRCC improvement: {srcc_no_tta:.5f} -> {srcc_tta:.5f} (Δ = {srcc_tta - srcc_no_tta:+.5f})")
    print("=" * 60)
