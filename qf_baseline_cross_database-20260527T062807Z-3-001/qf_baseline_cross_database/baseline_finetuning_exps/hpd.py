#### CODE FOR BASELINE Qformer with 10% image inputs for HPD
### HPD baseline with 10% labeled data only
##### Multimodal / image branch only
##### Pairwise supervision with BCEWithLogitsLoss on (score1 - score2)
##### Save best test predictions CSV

##### ----------------- ####
#####  1) IMPORTS + UTILS
##### ----------------- ####
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from lavis.models import load_model_and_preprocess


##### SET SEED #####
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEED = 1234
set_seed(SEED)


#### METRICS ####
def pairwise_accuracy_from_pairlogit(pair_logit: np.ndarray, pair_target: np.ndarray) -> float:
    pair_logit = np.asarray(pair_logit)
    pair_target = np.asarray(pair_target).astype(np.float32)
    if pair_logit.size == 0:
        return 0.0
    pair_pred = (pair_logit > 0).astype(np.float32)
    return float((pair_pred == pair_target).mean())


##### ----------------- ####
#####  2) DATASET + COLLATE
##### ----------------- ####
IMG1_COL   = "Image1"
IMG2_COL   = "Image2"
PROMPT_COL = "Prompt"
LABEL1_COL = "Label1"
LABEL2_COL = "Label2"
DESC1_COL  = "gen_response1"
DESC2_COL  = "gen_response2"


class QFormerPairwiseDataset(Dataset):
    """
    Pairwise dataset built from df or csv_path.
    Filters invalid rows at init (both images must exist and be readable).
    """
    def __init__(self, img_root: str, df: pd.DataFrame = None, csv_path: str = None, image_tf=None, verbose=True):
        assert (df is not None) ^ (csv_path is not None), "Provide exactly one of df or csv_path"
        self.df = df.copy() if df is not None else pd.read_csv(csv_path)
        self.img_root = img_root
        self.verbose = verbose

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

        valid_indices = []
        dropped = 0

        for i in range(len(self.df)):
            row = self.df.iloc[i]
            p1 = os.path.join(self.img_root, str(row[IMG1_COL]))
            p2 = os.path.join(self.img_root, str(row[IMG2_COL]))
            try:
                if (not os.path.isfile(p1)) or (not os.path.isfile(p2)):
                    dropped += 1
                    continue
                with Image.open(p1) as im1:
                    im1.verify()
                with Image.open(p2) as im2:
                    im2.verify()
                valid_indices.append(i)
            except Exception:
                dropped += 1
                continue

        self.df = self.df.iloc[valid_indices].reset_index(drop=True)

        if self.verbose:
            print(f"[QFormerPairwiseDataset] input={len(valid_indices) + dropped} | valid={len(self.df)} | dropped={dropped}")

    def __len__(self):
        return len(self.df)

    def _load_image_tensor(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return self.image_tf(img)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_name1 = str(row[IMG1_COL])
        image_name2 = str(row[IMG2_COL])

        image_path1 = os.path.join(self.img_root, image_name1)
        image_path2 = os.path.join(self.img_root, image_name2)

        image1 = self._load_image_tensor(image_path1)
        image2 = self._load_image_tensor(image_path2)

        prompt = str(row[PROMPT_COL])
        desc1 = str(row[DESC1_COL])
        desc2 = str(row[DESC2_COL])

        label1 = torch.tensor(float(row[LABEL1_COL]), dtype=torch.float32)
        label2 = torch.tensor(float(row[LABEL2_COL]), dtype=torch.float32)

        return {
            "image1": image1,
            "image2": image2,
            "prompt": prompt,
            "desc1": desc1,
            "desc2": desc2,
            "image_name1": image_name1,
            "image_name2": image_name2,
            "label1": label1,
            "label2": label2,
        }


def collate_fn(batch):
    return {
        "images1": torch.stack([b["image1"] for b in batch], dim=0),
        "images2": torch.stack([b["image2"] for b in batch], dim=0),
        "prompts": [b["prompt"] for b in batch],
        "descs1": [b["desc1"] for b in batch],
        "descs2": [b["desc2"] for b in batch],
        "image_names1": [b["image_name1"] for b in batch],
        "image_names2": [b["image_name2"] for b in batch],
        "label1": torch.stack([b["label1"] for b in batch], dim=0),
        "label2": torch.stack([b["label2"] for b in batch], dim=0),
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
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.layer(x)
# class Regressor(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim=1):
#         super().__init__()
#         self.layer = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, output_dim),
#         )

#     def forward(self, x):
#         return self.layer(x)

class QformerWrapper(nn.Module):
    """
    Only multimodal/image branch is used here.

    forward(images, prompts, descs) returns:
      mm_mean_embeds: [B, 768]

    descs is accepted only to keep interface similar to other scripts.
    """
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


##### ----------------- ####
#####  5) INIT + FREEZE/UNFREEZE
##### ----------------- ####
qformer = QformerWrapper(device=device, is_eval=False).to(device)
regressor = Regressor(input_dim=768, output_dim=1).to(device)
# regressor = Regressor(input_dim=768, hidden_dim=256, output_dim=1).to(device)


def load_checkpoint(checkpoint_path: str):
    """
    Load EvalMI-pretrained baseline checkpoint for mm-only model.
    Expected keys:
      - qformer.Qformer
      - query_tokens
      - regressor
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if "qformer.Qformer" not in ckpt:
        raise KeyError("Missing key 'qformer.Qformer' in checkpoint.")
    if "query_tokens" not in ckpt:
        raise KeyError("Missing key 'query_tokens' in checkpoint.")
    if "regressor" not in ckpt:
        raise KeyError("Missing key 'regressor' in checkpoint.")

    qformer.model.Qformer.load_state_dict(ckpt["qformer.Qformer"], strict=True)

    qt = ckpt["query_tokens"]
    if not isinstance(qt, torch.Tensor):
        raise TypeError(f"Expected query_tokens to be Tensor, got {type(qt)}")

    qt = qt.to(qformer.model.query_tokens.device, dtype=qformer.model.query_tokens.dtype)
    with torch.no_grad():
        qformer.model.query_tokens.copy_(qt)

    regressor.load_state_dict(ckpt["regressor"], strict=True)

    print(f"Loaded Qformer+query_tokens+regressor from: {checkpoint_path}")


# ---- load your EvalMI-trained ckpt ----
INIT_CKPT = "/storage/users/rajivs/anatapmitra/pretraining/evalmi_baseline_qf_ver1.pth"
load_checkpoint(INIT_CKPT)

# Freeze everything in BLIP2 feature extractor
for p in qformer.model.parameters():
    p.requires_grad = False

# Unfreeze ONLY query tokens + Qformer
qformer.model.query_tokens.requires_grad = True
for p in qformer.model.Qformer.parameters():
    p.requires_grad = True

for p in regressor.parameters():
    p.requires_grad = True


##### ----------------- ####
#####  6) LOSSES + OPTIM
##### ----------------- ####
reg_criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    [p for p in qformer.model.parameters() if p.requires_grad] + list(regressor.parameters()),
    lr=1e-5
)


#### TRAIN ONE EPOCH ####
def train_one_epoch(dataloader):
    qformer.train()
    regressor.train()

    total_loss = 0.0

    for batch in tqdm(dataloader, total=len(dataloader), desc="Training on HPD"):
        images1 = batch["images1"].to(device, non_blocking=True)
        images2 = batch["images2"].to(device, non_blocking=True)
        prompts = batch["prompts"]
        descs1 = batch["descs1"]
        descs2 = batch["descs2"]
        label1 = batch["label1"].to(device, non_blocking=True)
        label2 = batch["label2"].to(device, non_blocking=True)

        B = images1.size(0)

        # create one big batch of 2B images
        images = torch.cat([images1, images2], dim=0)
        prompts_2b = prompts + prompts
        descs_2b = descs1 + descs2

        optimizer.zero_grad(set_to_none=True)

        mm_mean_embeds = qformer(images, prompts_2b, descs_2b)
        pred = regressor(mm_mean_embeds).squeeze(-1)   # (2B,)
        pred1, pred2 = torch.split(pred, B, dim=0)

        pair_logit = pred1 - pred2
        pair_target = (label1 > label2).float()

        reg_loss = reg_criterion(pair_logit, pair_target)
        reg_loss.backward()
        optimizer.step()

        total_loss += float(reg_loss.detach().item())

    return total_loss / len(dataloader)


#### EVALUATE FUNCTION ####
@torch.no_grad()
def evaluate_and_save_df(dataloader, output_csv_path=None, desc_tag="Evaluating"):
    """
    Runs inference, computes pairwise accuracy, and returns a DataFrame with per-pair records:
      image_name1, image_name2, prompt, desc1, desc2, label1, label2,
      pred_score1, pred_score2, pair_logit, pair_target, pair_pred

    If output_csv_path is provided, saves the DataFrame to CSV.
    """
    qformer.eval()
    regressor.eval()

    rows = []
    pair_logit_list = []
    pair_target_list = []

    for batch in tqdm(dataloader, total=len(dataloader), desc=desc_tag):
        images1 = batch["images1"].to(device, non_blocking=True)
        images2 = batch["images2"].to(device, non_blocking=True)

        image_names1 = batch["image_names1"]
        image_names2 = batch["image_names2"]
        prompts = batch["prompts"]
        descs1 = batch["descs1"]
        descs2 = batch["descs2"]
        label1 = batch["label1"].to(device, non_blocking=True)
        label2 = batch["label2"].to(device, non_blocking=True)

        B = images1.size(0)

        images = torch.cat([images1, images2], dim=0)
        prompts_2b = prompts + prompts
        descs_2b = descs1 + descs2

        mm_mean_embeds = qformer(images, prompts_2b, descs_2b)
        pred = regressor(mm_mean_embeds).squeeze(-1)   # (2B,)
        pred1, pred2 = torch.split(pred, B, dim=0)

        pair_logit = pred1 - pred2
        pair_target = (label1 > label2).float()

        pair_logit_list.append(pair_logit.detach().float().cpu())
        pair_target_list.append(pair_target.detach().float().cpu())

        pred1_cpu = pred1.detach().float().cpu().numpy()
        pred2_cpu = pred2.detach().float().cpu().numpy()
        pair_logit_cpu = pair_logit.detach().float().cpu().numpy()
        pair_target_cpu = pair_target.detach().float().cpu().numpy()
        label1_cpu = label1.detach().float().cpu().numpy()
        label2_cpu = label2.detach().float().cpu().numpy()

        for i in range(B):
            pair_pred = 1.0 if pair_logit_cpu[i] > 0 else 0.0
            rows.append({
                "image_name1": image_names1[i],
                "image_name2": image_names2[i],
                "prompt": prompts[i],
                "gen_response1": descs1[i],
                "gen_response2": descs2[i],
                "label1": float(label1_cpu[i]),
                "label2": float(label2_cpu[i]),
                "pred_score1": float(pred1_cpu[i]),
                "pred_score2": float(pred2_cpu[i]),
                "pair_logit": float(pair_logit_cpu[i]),
                "pair_target": float(pair_target_cpu[i]),
                "pair_pred": float(pair_pred),
            })

    pair_logit_all = torch.cat(pair_logit_list, dim=0).numpy()
    pair_target_all = torch.cat(pair_target_list, dim=0).numpy()

    pair_acc = pairwise_accuracy_from_pairlogit(pair_logit_all, pair_target_all)

    df = pd.DataFrame(rows)

    if output_csv_path is not None:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        print(f"[Saved] Per-pair results CSV: {output_csv_path}")

    return pair_acc, df


#### MAIN PIPELINE ####
fraction = 0.1

train_csv = "/home/rajivs/anatapmitra/anatap_data/generated_descriptions_PT1/Hpd_new/Hpd_train_full_gen_responses_PT1_full.csv"
val_csv   = "/home/rajivs/anatapmitra/anatap_data/generated_descriptions_PT1/Hpd_new/Hpd_val_full_gen_responses_PT1_full.csv"
test_csv  = "/home/rajivs/anatapmitra/anatap_data/generated_descriptions_PT1/Hpd_new/Hpd_test_full_gen_responses_PT1_full.csv"
train_img_root  = "/home/rajivs/anatapmitra/anatap_data/hpd_data/train/train"
val_img_root = "/home/rajivs/anatapmitra/anatap_data/hpd_data/train/train"
test_img_root = "/home/rajivs/anatapmitra/anatap_data/hpd_data/test_images/test"

train_df = pd.read_csv(train_csv)
train_labeled_df = train_df.sample(frac=fraction, random_state=SEED).reset_index(drop=True)

test_out_csv = "/home/rajivs/anatapmitra/anatap_data/Qformer_experiments/SSL_exps/baseline/check_hpd_qf_bs_ft_v235.csv"

# datasets + loaders
train_data = QFormerPairwiseDataset(df=train_labeled_df, img_root=train_img_root, verbose=True)
train_loader = DataLoader(
    train_data,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True,
)

val_dataset = QFormerPairwiseDataset(csv_path=val_csv, img_root=val_img_root, verbose=True)
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True,
)

test_dataset = QFormerPairwiseDataset(csv_path=test_csv, img_root=test_img_root, verbose=True)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True,
)

print("length of train_data is:", len(train_data))
print("length of val_data is:", len(val_dataset))
print("length of test_data is:", len(test_dataset))

#### TRAINING + EVALUATION ####
no_epochs = 15
best_val_acc = -1.0
best_test_acc = -1.0

for epoch in range(no_epochs):
    train_loss = train_one_epoch(train_loader)
    print(f"Train loss at epoch {epoch + 1} is {train_loss}")

    val_acc, _ = evaluate_and_save_df(
        val_loader,
        output_csv_path=None,
        desc_tag="Evaluating on HPD validation",
    )
    print(f"Validation pairwise acc at epoch {epoch + 1} is {val_acc}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc

        test_acc, _ = evaluate_and_save_df(
            test_loader,
            output_csv_path=test_out_csv,
            desc_tag="Evaluating on HPD test",
        )
        print("Test evaluated")
        print(f"Test pairwise acc at epoch {epoch + 1} is {test_acc}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc

print(f"best test pairwise acc found to be {best_test_acc}")
print("Baseline QL on HPD")