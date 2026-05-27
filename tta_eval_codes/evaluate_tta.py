"""
Modular TTA evaluation for IQA datasets (A3K, A20K, EvalMI).

Usage:
    # Baseline (no TTA)
    python evaluate_tta.py --dataset a3k

    # Single TTA config
    python evaluate_tta.py --dataset a3k --losses gc rank --unfreeze both

    # FAGC + Adaptive Rank on query tokens only
    python evaluate_tta.py --dataset a20k --losses fagc adaptive_rank --unfreeze query

NOTE: Uses PRECOMPUTED ViT embeddings for the base forward pass.
      Rank-based losses additionally need raw images (loaded automatically).
"""

import os, sys, argparse, random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

from lavis.models import load_model_and_preprocess

# Ensure tta_framework is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tta_framework import LOSS_REGISTRY, TTAEngine
from tta_framework.param_strategy import print_param_summary

# ── Paths ──────────────────────────────────────────────────────────────────
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
_SPLIT_DIR   = os.path.join(
    _PROJECT_DIR,
    "important split files-20260527T062853Z-3-001",
    "important split files",
)

DEFAULT_CHECKPOINT = os.path.join(_PROJECT_DIR, "checkpoints", "evalmi_baseline_qf.pth")

# ── Dataset Configs ────────────────────────────────────────────────────────
DATASET_CONFIGS = {
    "a3k": {
        "splits": {
            "train": os.path.join(_SPLIT_DIR, "A3K_new", "a3k_train_full_gen_responses_PT1_normalized.csv"),
            "val":   os.path.join(_SPLIT_DIR, "A3K_new", "a3k_val_full_gen_responses_PT1_normalized.csv"),
            "test":  os.path.join(_SPLIT_DIR, "A3K_new", "a3k_test_full_gen_responses_PT1_normalized.csv"),
        },
        "embed_root": "/media/parth/021f75bf-bae8-49ef-86a5-28ca19171835/parth/dataset/embeddings/a3k",
        "img_root":   "/media/parth/021f75bf-bae8-49ef-86a5-28ca19171835/parth/agiqa-3k/images",
        "img_col": "image_name", "prompt_col": "prompt",
        "desc_col": "gen_answer", "gt_col": "gt_score",
        "embed_ext": ".npy", "embed_load": "npy",
        "img_subdir": False,   # flat directory
    },
    "a20k": {
        "splits": {
            "train": os.path.join(_SPLIT_DIR, "A20K_new", "A20k_train_full_PT1_normalized.csv"),
            "val":   os.path.join(_SPLIT_DIR, "A20K_new", "A20k_val_full_PT1_normalized.csv"),
            "test":  os.path.join(_SPLIT_DIR, "A20K_new", "A20k_test_full_PT1_normalized.csv"),
        },
        "embed_root": "/media/parth/021f75bf-bae8-49ef-86a5-28ca19171835/parth/dataset/embeddings/a20k",
        "img_root":   os.path.normpath(os.path.join(_SCRIPT_DIR, "../../agiqa-20k/AIGCQA-30K-Image")),
        "img_col": "image_name", "prompt_col": "prompt",
        "desc_col": "gen_answer", "gt_col": "gt_score",
        "embed_ext": ".npz", "embed_load": "npz",
        "img_subdir": True,    # train/val/test subdirs
    },
    "evalmi": {
        "splits": {
            "train": os.path.join(_SPLIT_DIR, "Evalmi_new", "evalmi_train_full_gen_responses_PT1.csv"),
            "val":   os.path.join(_SPLIT_DIR, "Evalmi_new", "evalmi_val_full_gen_responses_PT1.csv"),
            "test":  os.path.join(_SPLIT_DIR, "Evalmi_new", "evalmi_test_full_gen_responses_PT1.csv"),
        },
        "embed_root": "/media/parth/021f75bf-bae8-49ef-86a5-28ca19171835/parth/dataset/embeddings/evalmi",
        "img_root":   os.path.normpath(os.path.join(_PROJECT_DIR, "../EvalMi-50K/AIGI2025")),
        "img_col": "image_name", "prompt_col": "prompt",
        "desc_col": "gen_answer", "gt_col": "gt_score",
        "embed_ext": ".npz", "embed_load": "npz",
        "img_subdir": True,
    },
}

# ── Metrics ────────────────────────────────────────────────────────────────
def _rankdata(a):
    a = np.asarray(a); s = np.argsort(a); inv = np.empty_like(s)
    inv[s] = np.arange(len(a)); a_s = a[s]
    obs = np.concatenate(([True], a_s[1:] != a_s[:-1]))
    dr = np.cumsum(obs); c = np.cumsum(np.bincount(dr))
    return ((c[dr] + c[dr-1] + 1) / 2.0)[inv]

def spearmanr(x, y):
    x, y = np.asarray(x), np.asarray(y)
    rx, ry = _rankdata(x) - _rankdata(x).mean(), _rankdata(y) - _rankdata(y).mean()
    d = np.sqrt(np.sum(rx**2)*np.sum(ry**2))
    return float(np.sum(rx*ry)/d) if d else np.nan

def pearsonr(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    xm, ym = x-x.mean(), y-y.mean()
    d = np.sqrt(np.sum(xm**2)*np.sum(ym**2))
    return float(np.sum(xm*ym)/d) if d else np.nan

# ── Seed ───────────────────────────────────────────────────────────────────
def set_seed(s=1234):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ── VGG Feature Extractor ─────────────────────────────────────────────────
class VGGFeatureExtractor(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()
        self.device = device
    def forward(self, images):
        images = images.to(self.device)
        with torch.no_grad():
            f = self.vgg(images)
            f = F.adaptive_avg_pool2d(f, (1,1)).view(f.size(0), -1)
        return F.normalize(f, p=2, dim=-1)

# ── Projection Head (matches original TTA-IQA: Linear → Sigmoid) ─────────
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=768, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, output_dim), nn.Sigmoid())
    def forward(self, x):
        return self.net(x)

# ── Regressor ──────────────────────────────────────────────────────────────
class Regressor(nn.Module):
    def __init__(self, input_dim=768, output_dim=1):
        super().__init__()
        self.layer = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.layer(x)

# ── Q-Former Wrapper (supports both precomputed and raw-image modes) ──────
class QformerWrapper(nn.Module):
    def __init__(self, device, is_eval=True, keep_vit=False):
        super().__init__()
        model, _, _ = load_model_and_preprocess(
            name="blip2_feature_extractor", model_type="pretrain",
            is_eval=is_eval, device=device,
        )
        self.model = model.to(device)
        self.device = device
        if not keep_vit:
            del self.model.visual_encoder, self.model.ln_vision
            torch.cuda.empty_cache()

    def extract_image_embeds(self, images):
        images = images.to(self.device)
        with torch.no_grad():
            with self.model.maybe_autocast():
                return self.model.ln_vision(self.model.visual_encoder(images)).float()

    def forward_qformer(self, image_embeds_frozen, prompts, descs):
        B = image_embeds_frozen.size(0)
        image_embeds_frozen = image_embeds_frozen.to(self.device)
        image_atts = torch.ones(image_embeds_frozen.size()[:-1], dtype=torch.long, device=self.device)
        query_tokens = self.model.query_tokens.expand(B, -1, -1)
        text_prompt = self.model.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=self.device)
        mm_mask = torch.cat([query_atts, text_prompt.attention_mask], dim=1)
        mm_out = self.model.Qformer.bert(
            text_prompt.input_ids, query_embeds=query_tokens, attention_mask=mm_mask,
            encoder_hidden_states=image_embeds_frozen, encoder_attention_mask=image_atts, return_dict=True,
        )
        return mm_out.last_hidden_state[:, :query_tokens.size(1), :].mean(dim=1)

    def forward(self, images, prompts, descs):
        return self.forward_qformer(self.extract_image_embeds(images), prompts, descs)

# ── Dataset ────────────────────────────────────────────────────────────────
class TTADataset(Dataset):
    """Loads precomputed embeds + optionally raw images for augmentation."""
    def __init__(self, csv_path, cfg, load_raw_images=False):
        self.df = pd.read_csv(csv_path)
        self.df.columns = self.df.columns.str.strip()
        self.cfg = cfg
        self.load_raw = load_raw_images

        # Build image lookup for raw images
        if self.load_raw:
            self.img_lookup = self._build_lookup(cfg["img_root"], cfg["img_subdir"])
            self.clip_tf = transforms.Compose([
                transforms.Resize((224,224), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.48145466,0.4578275,0.40821073), std=(0.26862954,0.26130258,0.27577711)),
            ])
            self.vgg_tf = transforms.Compose([
                transforms.Resize((224,224), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])

    @staticmethod
    def _build_lookup(img_root, has_subdirs):
        lookup = {}
        if has_subdirs:
            for sd in os.listdir(img_root):
                sp = os.path.join(img_root, sd)
                if not os.path.isdir(sp): continue
                for f in os.listdir(sp):
                    if f.lower().endswith(('.png','.jpg','.jpeg')):
                        lookup[f] = os.path.join(sp, f)
                        # Also store with subdir prefix for EvalMI
                        lookup[os.path.join(sd, f)] = os.path.join(sp, f)
        else:
            for f in os.listdir(img_root):
                if f.lower().endswith(('.png','.jpg','.jpeg')):
                    lookup[f] = os.path.join(img_root, f)
        return lookup

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row[self.cfg["img_col"]])

        # Load precomputed embedding
        base = img_name.replace(".png", self.cfg["embed_ext"]).replace(".jpg", self.cfg["embed_ext"])
        embed_path = os.path.join(self.cfg["embed_root"], base)
        if self.cfg["embed_load"] == "npy":
            image_embeds = torch.from_numpy(np.load(embed_path)).float()
        else:
            image_embeds = torch.from_numpy(np.load(embed_path)["embed"]).float()

        prompt = str(row[self.cfg["prompt_col"]])
        desc = str(row.get(self.cfg["desc_col"], ""))
        gt = torch.tensor(float(row[self.cfg["gt_col"]]), dtype=torch.float32)

        out = {
            "image_embeds": image_embeds,
            "prompt": prompt, "description": desc,
            "image_name": img_name, "gt_score": gt,
        }

        if self.load_raw:
            path = self.img_lookup.get(img_name)
            if path is None:
                raise FileNotFoundError(f"Raw image not found: {img_name}")
            img = Image.open(path).convert("RGB")
            out["clip_image"] = self.clip_tf(img)
            out["vgg_image"] = self.vgg_tf(img)

        return out

def collate_fn(batch):
    out = {
        "image_embeds": torch.stack([b["image_embeds"] for b in batch]),
        "prompts": [b["prompt"] for b in batch],
        "descs": [b["description"] for b in batch],
        "image_names": [b["image_name"] for b in batch],
        "gt_scores": torch.stack([b["gt_score"] for b in batch]),
    }
    if "clip_image" in batch[0]:
        out["clip_images"] = torch.stack([b["clip_image"] for b in batch])
        out["vgg_images"] = torch.stack([b["vgg_image"] for b in batch])
    return out

# ── Evaluation ─────────────────────────────────────────────────────────────
def run_evaluation(engine, dataloader, device, desc_tag="Eval"):
    """Run TTA evaluation over all batches. Returns preds, gts, per-image df."""
    all_preds, all_gts, all_meta = [], [], []
    for batch in tqdm(dataloader, desc=desc_tag):
        preds, gts, meta = engine.adapt_and_predict(batch)
        all_preds.append(preds)
        all_gts.append(gts)
        all_meta.extend(meta)
    preds = np.concatenate(all_preds)
    gts = np.concatenate(all_gts)
    return preds, gts, pd.DataFrame(all_meta)

@torch.no_grad()
def run_baseline(qformer, regressor, dataloader, device, desc_tag="Baseline"):
    """Run baseline (no TTA) evaluation."""
    qformer.eval(); regressor.eval()
    all_preds, all_gts, rows = [], [], []
    for batch in tqdm(dataloader, desc=desc_tag):
        embeds = batch["image_embeds"].to(device, non_blocking=True)
        gt = batch["gt_scores"].to(device, non_blocking=True)
        mm = qformer.forward_qformer(embeds, batch["prompts"], batch["descs"])
        pred = regressor(mm).squeeze(-1)
        p, g = pred.float().cpu().numpy(), gt.float().cpu().numpy()
        all_preds.append(p); all_gts.append(g)
        for i in range(len(batch["image_names"])):
            rows.append({"image_name": batch["image_names"][i], "prompt": batch["prompts"][i],
                         "gen_answer": batch["descs"][i], "gt_score": float(g[i]), "pred_score": float(p[i])})
    return np.concatenate(all_preds), np.concatenate(all_gts), pd.DataFrame(rows)

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Modular TTA Evaluation for IQA")
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--losses", nargs="*", default=[], choices=list(LOSS_REGISTRY.keys()),
                        help="TTA losses to apply. Empty = baseline only.")
    parser.add_argument("--unfreeze", type=str, default="layernorm", choices=["none", "layernorm", "query", "both"])
    parser.add_argument("--tta_steps", type=int, default=3)
    parser.add_argument("--tta_lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for contrastive losses")
    parser.add_argument("--freeze_proj_head", action="store_true", default=True,
                        help="Keep projection head frozen during TTA (matches original)")
    parser.add_argument("--update_proj_head", dest="freeze_proj_head", action="store_false",
                        help="Allow projection head to be updated during TTA")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = DATASET_CONFIGS[args.dataset]
    loss_names = args.losses
    use_tta = len(loss_names) > 0

    # Determine what resources are needed
    needs_aug = any(LOSS_REGISTRY[n].requires_augmentations for n in loss_names)
    needs_vgg = any(LOSS_REGISTRY[n].requires_vgg for n in loss_names)
    needs_raw = needs_aug or needs_vgg
    keep_vit = needs_aug  # ViT needed only for augmented embedding extraction

    # Output dir
    if args.output_dir is None:
        loss_tag = "_".join(loss_names) if loss_names else "baseline"
        args.output_dir = os.path.join(_SCRIPT_DIR, "results",
            f"{args.dataset}_tta", f"{loss_tag}__{args.unfreeze}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Config summary
    print("=" * 60)
    print(f"  Dataset:    {args.dataset}")
    print(f"  Split:      {args.split}")
    print(f"  Losses:     {loss_names if loss_names else '(baseline, no TTA)'}")
    print(f"  Unfreeze:   {args.unfreeze}")
    print(f"  TTA steps:  {args.tta_steps}")
    print(f"  TTA LR:     {args.tta_lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Output:     {args.output_dir}")
    print("=" * 60)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load models ──
    print("Loading Q-Former model...")
    qformer = QformerWrapper(device=device, is_eval=True, keep_vit=keep_vit).to(device)
    regressor = Regressor(input_dim=768, output_dim=1).to(device)
    proj_head = ProjectionHead(input_dim=768, output_dim=128).to(device)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    qformer.model.Qformer.load_state_dict(ckpt["qformer.Qformer"], strict=True)
    qformer.model.query_tokens = nn.Parameter(ckpt["query_tokens"].to(device))
    regressor.load_state_dict(ckpt["regressor"], strict=True)
    if "proj_head" in ckpt:
        proj_head.load_state_dict(ckpt["proj_head"])
    print("Checkpoint loaded.")

    # ── VGG (only if needed) ──
    vgg = VGGFeatureExtractor(device) if needs_vgg else None

    # ── Datasets ──
    if args.split == "all":
        split_csvs = cfg["splits"]
    else:
        split_csvs = {args.split: cfg["splits"][args.split]}

    # ── Evaluate each split ──
    summary_rows = []
    combined_preds, combined_gts, combined_dfs = [], [], []

    for split_name, csv_path in split_csvs.items():
        print(f"\n{'='*60}")
        print(f"  Split: {split_name.upper()}  |  CSV: {csv_path}")
        print(f"{'='*60}")

        ds = TTADataset(csv_path, cfg, load_raw_images=needs_raw)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
        print(f"  Samples: {len(ds)}")

        # ── Baseline ──
        preds_base, gts_base, df_base = run_baseline(
            qformer, regressor, loader, device, desc_tag=f"Baseline {split_name}")
        srcc_base = spearmanr(preds_base, gts_base)
        plcc_base = pearsonr(preds_base, gts_base)
        print(f"  [Baseline]  SRCC: {srcc_base:.6f}  |  PLCC: {plcc_base:.6f}")

        df_base.insert(0, "split", split_name)
        base_csv = os.path.join(args.output_dir, f"{args.dataset}_{split_name}_baseline.csv")
        df_base.to_csv(base_csv, index=False)

        summary_rows.append({"split": split_name, "mode": "baseline",
                             "n": len(ds), "srcc": srcc_base, "plcc": plcc_base})

        # ── TTA ──
        if use_tta:
            # Instantiate losses
            losses = []
            for ln in loss_names:
                cls = LOSS_REGISTRY[ln]
                if ln in ("gc", "fagc"):
                    losses.append(cls(temperature=args.temperature))
                else:
                    losses.append(cls())

            if split_name == list(split_csvs.keys())[0]:
                print_param_summary(qformer, args.unfreeze)

            engine = TTAEngine(
                qformer=qformer, regressor=regressor, proj_head=proj_head,
                losses=losses, unfreeze_strategy=args.unfreeze,
                tta_steps=args.tta_steps, tta_lr=args.tta_lr,
                freeze_proj_head=args.freeze_proj_head,
                vgg_extractor=vgg, device=device,
            )

            preds_tta, gts_tta, df_tta = run_evaluation(
                engine, loader, device, desc_tag=f"TTA {split_name}")
            srcc_tta = spearmanr(preds_tta, gts_tta)
            plcc_tta = pearsonr(preds_tta, gts_tta)
            print(f"  [TTA]       SRCC: {srcc_tta:.6f}  |  PLCC: {plcc_tta:.6f}")

            df_tta.insert(0, "split", split_name)
            tta_csv = os.path.join(args.output_dir, f"{args.dataset}_{split_name}_tta.csv")
            df_tta.to_csv(tta_csv, index=False)

            summary_rows.append({"split": split_name, "mode": "tta",
                                 "n": len(ds), "srcc": srcc_tta, "plcc": plcc_tta})
            combined_preds.append(preds_tta); combined_gts.append(gts_tta)
            combined_dfs.append(df_tta)
        else:
            combined_preds.append(preds_base); combined_gts.append(gts_base)
            combined_dfs.append(df_base)

    # ── Combined metrics ──
    if len(combined_preds) > 1:
        cp = np.concatenate(combined_preds); cg = np.concatenate(combined_gts)
        summary_rows.append({"split": "combined", "mode": "tta" if use_tta else "baseline",
                             "n": len(cp), "srcc": spearmanr(cp, cg), "plcc": pearsonr(cp, cg)})

    # ── Summary ──
    print("\n" + "=" * 60)
    loss_tag = "+".join(loss_names) if loss_names else "none"
    print(f"  RESULTS — {args.dataset.upper()} | Losses: {loss_tag} | Unfreeze: {args.unfreeze}")
    print("=" * 60)
    print(f"  {'Split':<10} {'Mode':<10} {'N':>8} {'SRCC':>10} {'PLCC':>10}")
    print("-" * 60)
    for r in summary_rows:
        print(f"  {r['split']:<10} {r['mode']:<10} {r['n']:>8} {r['srcc']:>10.6f} {r['plcc']:>10.6f}")
    print("=" * 60)

    summary_csv = os.path.join(args.output_dir, f"{args.dataset}_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"\n[Saved] Summary: {summary_csv}")
    print(f"[Saved] Per-image CSVs in: {args.output_dir}")

if __name__ == "__main__":
    set_seed(1234)
    main()
