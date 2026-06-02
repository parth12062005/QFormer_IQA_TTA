import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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
DEFAULT_TEST_CSV   = "/home/rajivs/anatapmitra/anatap_data/generated_descriptions_PT1/A20K_new/A20k_test_full_PT1_normalized.csv"
DEFAULT_IMG_ROOT   = "/home/rajivs/anatapmitra/anatap_data/anatap_1/aigiqa-20k/Images/train"

IMG_COL    = "image_name"
PROMPT_COL = "prompt"
DESC_COL   = "gen_answer"
GT_COL     = "gt_score"

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
class A20KDataset(Dataset):
    def __init__(self, csv_path, img_root):
        self.df = pd.read_csv(csv_path)
        self.img_root = img_root
        
        self.image_tf = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])

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
            "image": image,
            "prompt": prompt,
            "description": desc,
            "image_name": str(row[IMG_COL]),
            "gt_score": gt,
        }

def collate_fn(batch):
    return {
        "images": torch.stack([b["image"] for b in batch], dim=0),
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
        # Using single-layer nn.Linear to match evalmi_baseline_qf.pth
        self.layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.layer(x)

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


##### ------------- ####
#####  MAIN
##### ------------- ####
def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline Q-Former checkpoint on A20K")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test_csv", type=str, default=DEFAULT_TEST_CSV)
    parser.add_argument("--img_root", type=str, default=DEFAULT_IMG_ROOT)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    print(f"Device: {device}")

    print("Loading models...")
    qformer   = QformerWrapper(device=device, is_eval=True).to(device)
    regressor = Regressor(input_dim=768, output_dim=1).to(device)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    qformer.model.Qformer.load_state_dict(ckpt["qformer.Qformer"], strict=True)
    qformer.model.query_tokens = nn.Parameter(ckpt["query_tokens"].to(device))
    regressor.load_state_dict(ckpt["regressor"], strict=True)
    print("Checkpoint loaded successfully.")

    qformer.eval()
    regressor.eval()

    test_dataset = A20KDataset(csv_path=args.test_csv, img_root=args.img_root)
    test_loader  = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Test samples: {len(test_dataset)}")

    print("\n--- Evaluation on A20K Test Set ---")
    rows, pred_list, gt_list = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Eval A20K"):
            images = batch["images"]
            prompts = batch["prompts"]
            descs = batch["descs"]

            mm_embeds = qformer(images, prompts, descs)
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

    print("\n" + "=" * 50)
    print("  EVALUATION RESULTS (A20K Test)")
    print("=" * 50)
    print(f"  SRCC  : {srcc:.6f}")
    print(f"  PLCC  : {plcc:.6f}")
    print("=" * 50)

if __name__ == "__main__":
    set_seed(1234)
    main()
