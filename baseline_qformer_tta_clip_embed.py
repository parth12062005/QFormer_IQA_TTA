##### TTA Script: Feature Affinity-based Group Contrastive (FAGC) Loss
##
#### Test-Time Adaptation for baseline Q-Former IQA
#### Only query_tokens (32x768) + projection head are updated via backprop
####
#### Pipeline (episodic, per-batch):
####   1. Save original query_tokens
####   2. For each test batch:
####      a. Cluster using CLIP feature affinity (instead of VGG)
####      b. Compute FAGC loss in projection space
####      c. Backprop to update query_tokens + projection head
####      d. Predict quality scores with adapted model
####      e. Reset query_tokens for next batch

import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from torchvision import transforms

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
    return np.nan if denom == 0 else float(np.sum(rx * ry) / denom)

def pearsonr_numpy(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    assert x.shape == y.shape
    xm, ym = x - x.mean(), y - y.mean()
    denom = np.sqrt(np.sum(xm**2) * np.sum(ym**2))
    return np.nan if denom == 0 else float(np.sum(xm * ym) / denom)


##### ----------------- ####
#####  2) CONFIG
##### ----------------- ####
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

IMG_COL    = "image_name"
PROMPT_COL = "prompt"
DESC_COL   = "gen_answer"
GT_COL     = "gt_score"

EMBED_ROOT  = "/media/parth/Balance/parth/dataset/embeddings"
IMG_ROOT    = os.path.normpath(os.path.join(_SCRIPT_DIR, "../EvalMi-50K/AIGI2025"))
TRAIN_CSV   = os.path.normpath(os.path.join(_SCRIPT_DIR, "../EvalMi-50K/evalmi_train.csv"))
TEST_CSV    = os.path.normpath(os.path.join(_SCRIPT_DIR, "../EvalMi-50K/evalmi_test.csv"))
CHECKPOINT  = os.path.join(_SCRIPT_DIR, "checkpoints", "evalmi_baseline_qf.pth")
RESULTS_DIR = os.path.join(_SCRIPT_DIR, "results_clip_embed")


##### ----------------- ####
#####  3) DATASET
##### ----------------- ####
def _build_case_map(img_root):
    case_map = {}
    for entry in os.listdir(img_root):
        if os.path.isdir(os.path.join(img_root, entry)):
            case_map[entry.lower()] = entry
    return case_map

_CASE_MAP = _build_case_map(IMG_ROOT)

def _resolve_image_path(img_root, name):
    parts = name.split("/")
    if len(parts) >= 2:
        parts[0] = _CASE_MAP.get(parts[0].lower(), parts[0])
    return os.path.join(img_root, *parts)

# CLIP normalization
CLIP_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    ),
])

class TTADataset(Dataset):
    def __init__(self, csv_path, embed_root=EMBED_ROOT, img_root=IMG_ROOT):
        self.df = pd.read_csv(csv_path)
        self.embed_root = embed_root
        self.img_root = img_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row[IMG_COL])

        embed_path = os.path.join(
            self.embed_root,
            img_name.replace(".png", ".npz").replace(".jpg", ".npz"),
        )
        image_embeds = torch.from_numpy(np.load(embed_path)["embed"]).float()

        raw_path = _resolve_image_path(self.img_root, img_name)
        raw_image = CLIP_TRANSFORM(Image.open(raw_path).convert("RGB"))

        prompt = str(row[PROMPT_COL])
        desc   = str(row[DESC_COL])
        gt     = torch.tensor(float(row[GT_COL]), dtype=torch.float32)

        return {
            "image_embeds": image_embeds,
            "raw_image": raw_image,
            "prompt": prompt,
            "description": desc,
            "image_name": img_name,
            "gt_score": gt,
        }

def collate_fn(batch):
    return {
        "image_embeds": torch.stack([b["image_embeds"] for b in batch], dim=0),
        "raw_images":   torch.stack([b["raw_image"] for b in batch], dim=0),
        "prompts":      [b["prompt"] for b in batch],
        "descs":        [b["description"] for b in batch],
        "image_names":  [b["image_name"] for b in batch],
        "gt_scores":    torch.stack([b["gt_score"] for b in batch], dim=0),
    }


##### ----------------- ####
#####  4) MODELS
##### ----------------- ####
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
    def __init__(self, device, is_eval=False):
        super().__init__()
        model, _, _ = load_model_and_preprocess(
            name="blip2_feature_extractor",
            model_type="pretrain",
            is_eval=is_eval,
            device=device,
        )
        self.model = model.to(device)
        self.device = device

        del self.model.visual_encoder
        del self.model.ln_vision
        torch.cuda.empty_cache()

    def forward(self, image_embeds_frozen, prompts, descs):
        B = image_embeds_frozen.size(0)
        image_embeds_frozen = image_embeds_frozen.to(self.device)

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


##### ----------------- ####
#####  5) CLIP FEATURE EXTRACTOR
##### ----------------- ####
class CLIPFeatureExtractor(nn.Module):
    def __init__(self, device, model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        from transformers import CLIPVisionModelWithProjection
        self.model = CLIPVisionModelWithProjection.from_pretrained(model_name)
        self.to(device)
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        """x: [B,3,224,224] CLIP-normalized."""
        outputs = self.model(x)
        embeds = outputs.image_embeds
        return F.normalize(embeds, dim=-1)


##### ----------------- ####
#####  6) FAGC LOSS
##### ----------------- ####
def compute_fagc_loss(proj_feats, c1_idx, c2_idx, temperature=0.5):
    loss = torch.tensor(0.0, device=proj_feats.device)
    count = 0

    for cluster_self, cluster_other in [(c1_idx, c2_idx), (c2_idx, c1_idx)]:
        if len(cluster_self) < 2 or len(cluster_other) < 1:
            continue
        for i in cluster_self:
            pos_indices = [j for j in cluster_self if j != i]
            if len(pos_indices) == 0:
                continue

            pos_sims = torch.stack([
                F.cosine_similarity(proj_feats[i].unsqueeze(0), proj_feats[j].unsqueeze(0))
                for j in pos_indices
            ]).squeeze(-1)
            pos_logit = (pos_sims / temperature).logsumexp(0) - np.log(len(pos_indices))

            neg_sims = torch.stack([
                F.cosine_similarity(proj_feats[i].unsqueeze(0), proj_feats[k].unsqueeze(0))
                for k in cluster_other
            ]).squeeze(-1)
            neg_logits = neg_sims / temperature

            denom = torch.cat([pos_logit.unsqueeze(0), neg_logits])
            loss += -pos_logit + torch.logsumexp(denom, dim=0)
            count += 1

    return loss / max(count, 1)

def cluster_by_clip_affinity(clip_feats, anchor_idx):
    B = clip_feats.size(0)
    if B < 2:
        return list(range(B)), []

    anchor_feat = clip_feats[anchor_idx].unsqueeze(0)
    affinities = F.cosine_similarity(anchor_feat, clip_feats, dim=-1)

    sorted_indices = torch.argsort(affinities, descending=True).cpu().tolist()
    half = B // 2
    c1_idx = sorted_indices[:half]
    c2_idx = sorted_indices[half:]

    if anchor_idx in c2_idx:
        c2_idx.remove(anchor_idx)
        c1_idx.append(anchor_idx)

    return c1_idx, c2_idx


##### ----------------- ####
#####  7) INIT MODELS + LOAD CHECKPOINT
##### ----------------- ####
device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
print("device:", device)

qformer   = QformerWrapper(device=device, is_eval=False).to(device)
regressor = Regressor(input_dim=768, output_dim=1).to(device)
proj_head = ProjectionHead(input_dim=768, hidden_dim=256, output_dim=128).to(device)
clip_extractor = CLIPFeatureExtractor(device=device)

print(f"Loading checkpoint: {CHECKPOINT}")
ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
qformer.model.Qformer.load_state_dict(ckpt["qformer.Qformer"], strict=True)
qformer.model.query_tokens = nn.Parameter(ckpt["query_tokens"].to(device))
regressor.load_state_dict(ckpt["regressor"], strict=True)
print("Checkpoint loaded successfully.")

for p in qformer.model.parameters():
    p.requires_grad = False
for p in regressor.parameters():
    p.requires_grad = False

qformer.model.query_tokens.requires_grad = True
for p in proj_head.parameters():
    p.requires_grad = True

ORIGINAL_QUERY_TOKENS = qformer.model.query_tokens.detach().clone()


##### ----------------- ####
#####  8) TTA STEP
##### ----------------- ####
TTA_STEPS = 2
TTA_LR    = 1e-4
TAU       = 0.5

def tta_adapt_batch(image_embeds, raw_images, prompts, descs):
    with torch.no_grad():
        qformer.model.query_tokens.copy_(ORIGINAL_QUERY_TOKENS)
    for layer in proj_head.net:
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

    tta_optimizer = torch.optim.Adam(
        [qformer.model.query_tokens] + list(proj_head.parameters()),
        lr=TTA_LR,
    )

    image_embeds = image_embeds.to(device)
    raw_images = raw_images.to(device)

    # CLIP features for clustering
    clip_feats = clip_extractor(raw_images)

    with torch.no_grad():
        qformer.eval()
        regressor.eval()
        init_embeds = qformer(image_embeds, prompts, descs)
        init_scores = regressor(init_embeds).squeeze(-1)
        anchor_idx = int(torch.argmax(init_scores).item())

    # Cluster by CLIP affinity
    c1_idx, c2_idx = cluster_by_clip_affinity(clip_feats, anchor_idx)

    if len(c1_idx) < 2 or len(c2_idx) < 1:
        with torch.no_grad():
            qformer.eval()
            return qformer(image_embeds, prompts, descs)

    qformer.train()
    proj_head.train()

    for step in range(TTA_STEPS):
        tta_optimizer.zero_grad()
        mm_mean_embeds = qformer(image_embeds, prompts, descs)
        proj_feats = proj_head(mm_mean_embeds)
        fagc_loss = compute_fagc_loss(proj_feats, c1_idx, c2_idx, temperature=TAU)
        fagc_loss.backward()
        tta_optimizer.step()

    qformer.eval()
    with torch.no_grad():
        adapted_embeds = qformer(image_embeds, prompts, descs)
    return adapted_embeds


##### ----------------- ####
#####  9) EVALUATE FUNCTIONS
##### ----------------- ####
@torch.no_grad()
def evaluate_no_tta(dataloader, desc_tag="Eval (no TTA)"):
    qformer.eval()
    regressor.eval()
    qformer.model.query_tokens.copy_(ORIGINAL_QUERY_TOKENS)

    rows, pred_list, gt_list = [], [], []
    for batch in tqdm(dataloader, desc=desc_tag):
        image_embeds = batch["image_embeds"].to(device)
        mm_embeds = qformer(image_embeds, batch["prompts"], batch["descs"])
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
    return srcc, plcc, pd.DataFrame(rows)


def evaluate_with_tta(dataloader, desc_tag="Eval (FAGC TTA)"):
    regressor.eval()

    rows, pred_list, gt_list = [], [], []
    for batch in tqdm(dataloader, desc=desc_tag):
        image_embeds = batch["image_embeds"]
        raw_images   = batch["raw_images"]
        prompts      = batch["prompts"]
        descs        = batch["descs"]

        adapted_embeds = tta_adapt_batch(image_embeds, raw_images, prompts, descs)

        with torch.no_grad():
            pred = regressor(adapted_embeds).squeeze(-1)

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
    return srcc, plcc, pd.DataFrame(rows)


##### ----------------- ####
#####  10) MAIN
##### ----------------- ####
if __name__ == "__main__":
    print("=" * 60)
    print("FAGC TTA FOR BASELINE QFORMER (EvalMi-50K) - CLIP Embeds")
    print("Adapting: query_tokens only | Loss: FAGC | Episodic reset")
    print(f"TTA_STEPS={TTA_STEPS}  TTA_LR={TTA_LR}  TAU={TAU}")
    print("=" * 60)

    BATCH_SIZE  = 8
    NUM_WORKERS = 4

    train_dataset = TTADataset(csv_path=TRAIN_CSV)
    train_loader  = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True,
    )

    test_dataset = TTADataset(csv_path=TEST_CSV)
    test_loader  = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True,
    )

    print(f"Train set: {len(train_dataset)} samples")
    print(f"Test set:  {len(test_dataset)} samples")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n--- Evaluation WITHOUT TTA ---")
    train_srcc_no, train_plcc_no, train_df_no = evaluate_no_tta(train_loader, desc_tag="Train (no TTA)")
    test_srcc_no,  test_plcc_no,  test_df_no  = evaluate_no_tta(test_loader,  desc_tag="Test (no TTA)")

    print(f"\n  Train  SRCC={train_srcc_no:.6f}  PLCC={train_plcc_no:.6f}")
    print(f"  Test   SRCC={test_srcc_no:.6f}  PLCC={test_plcc_no:.6f}")

    print("\n--- Evaluation WITH FAGC TTA ---")
    train_srcc_tta, train_plcc_tta, train_df_tta = evaluate_with_tta(train_loader, desc_tag="Train (FAGC TTA)")
    test_srcc_tta,  test_plcc_tta,  test_df_tta  = evaluate_with_tta(test_loader,  desc_tag="Test (FAGC TTA)")

    print(f"\n  Train  SRCC={train_srcc_tta:.6f}  PLCC={train_plcc_tta:.6f}")
    print(f"  Test   SRCC={test_srcc_tta:.6f}  PLCC={test_plcc_tta:.6f}")

    for split_name, df_no, df_tta in [("train", train_df_no, train_df_tta),
                                       ("test",  test_df_no,  test_df_tta)]:
        merged = df_no[["image_name", "gt_score"]].copy()
        merged["pred_no_tta"] = df_no["pred_score"]
        merged["pred_tta"]    = df_tta["pred_score"]
        out_path = os.path.join(RESULTS_DIR, f"{split_name}_results.csv")
        merged.to_csv(out_path, index=False)
        print(f"[Saved] {out_path}")

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"{'':>8} {'SRCC (no TTA)':>14} {'SRCC (TTA)':>12} {'Δ SRCC':>10} {'PLCC (no TTA)':>14} {'PLCC (TTA)':>12} {'Δ PLCC':>10}")
    print("-" * 82)
    print(f"{'Train':>8} {train_srcc_no:>14.6f} {train_srcc_tta:>12.6f} {train_srcc_tta-train_srcc_no:>+10.6f} {train_plcc_no:>14.6f} {train_plcc_tta:>12.6f} {train_plcc_tta-train_plcc_no:>+10.6f}")
    print(f"{'Test':>8} {test_srcc_no:>14.6f} {test_srcc_tta:>12.6f} {test_srcc_tta-test_srcc_no:>+10.6f} {test_plcc_no:>14.6f} {test_plcc_tta:>12.6f} {test_plcc_tta-test_plcc_no:>+10.6f}")
    print("=" * 60)
