### script for running baseline again with 10 % data with new pretrained checkpoint
##### Script is to pretrain the baseline qformer with EvalMI 
##
#### script for FULL LABEL with baseline qformer where we only use mm pass only 
#### - Supervised: regression (multimodal query tokens pooled -> MLP -> MOS)
#### We save the predictions for test set only here .

##### ----------------- ####
#####  1) IMPORTS + UTILS
##### ----------------- ####
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
# ----------------------------
# NumPy-only implementation
# ----------------------------
def rankdata_numpy(a):
    """
    NumPy-only equivalent of scipy.stats.rankdata(method="average")
    """
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
    x = np.asarray(x)
    y = np.asarray(y)
    assert x.shape == y.shape, "x and y must have same shape"

    rx = rankdata_numpy(x)
    ry = rankdata_numpy(y)

    rx = rx - rx.mean()
    ry = ry - ry.mean()

    denom = np.sqrt(np.sum(rx**2) * np.sum(ry**2))
    if denom == 0:
        return np.nan
    return np.sum(rx * ry) / denom


##### ----------------- ####
#####  2) DATASET + COLLATE
##### ----------------- ####
IMG_COL    = "image_name"
PROMPT_COL = "prompt"
DESC_COL   = "gen_answer"   # change if your csv uses a different name
GT_COL     = "gt_score"

class QFormerQualityDataset(Dataset):
    """
    Can be constructed with either:
      - csv_path (str)  OR
      - df (pd.DataFrame)
    """
    def __init__(self, img_root, csv_path=None, df=None, image_tf=None):
        assert (csv_path is not None) ^ (df is not None), "Provide exactly one of csv_path or df"
        self.df = pd.read_csv(csv_path) if csv_path is not None else df.reset_index(drop=True)
        self.df.columns = self.df.columns.str.strip()
        self.img_root = img_root
        self.df.columns 

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
            "image": image,
            "prompt": prompt,
            "description": desc,
            "image_name": str(row[IMG_COL]),
            "gt_score": gt,
        }


def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)
    prompts = [b["prompt"] for b in batch]
    descs = [b["description"] for b in batch]
    image_names = [b["image_name"] for b in batch]
    gt_scores = torch.stack([b["gt_score"] for b in batch], dim=0)
    return {
        "images": images,
        "prompts": prompts,
        "descs": descs,
        "image_names": image_names,
        "gt_scores": gt_scores,
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
        """
        1 forward passes through SAME BLIP2-QFormer:

        (1) image-only query pass:
            images -> (frozen ViT) -> image_embeds_frozen

        (2) multimodal pass (for supervised regression):
            (images + prompts) -> Qformer(multimodal) -> query token outputs -> mean pool -> mm_mean [B,768]
        """
        B = images.size(0)
        images = images.to(self.device)

        # -------------------------
        # (0) IMAGE-EMBED COMPUTE
        # -------------------------
        with torch.no_grad():
            with self.model.maybe_autocast():
                image_embeds_frozen = self.model.ln_vision(self.model.visual_encoder(images))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long, device=self.device
            )

        query_tokens = self.model.query_tokens.expand(B, -1, -1)  # [B,32,768]
        # -------------------------
        # (1) MULTIMODAL PASS (PROMPTS)
        # -------------------------
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
#####  5) INIT + FREEZE/UNFREEZE
##### ----------------- ####
qformer = QformerWrapper(device=device, is_eval=False).to(device)
# regressor = Regressor(input_dim=768, output_dim=1).to(device)
regressor = Regressor(input_dim=768, output_dim=1).to(device)

#### LOAD CHECKPOINT EXACTLY FOR THIS MODEL ####
def load_checkpoint(checkpoint_path: str):
    """ This function loads the exact checkpoints that was saved during the evalmi pretraining"""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    
    ## Chcek key existence 
    if "qformer.Qformer" not in ckpt:
        raise KeyError("Missing key 'qformer.Qformer' in checkpoint.")
    if "query_tokens" not in ckpt:
        raise KeyError("Missing key 'query_tokens' in checkpoint.")
    if "regressor" not in ckpt:
        raise KeyError("Missing key 'regressor' in checkpoint")
    
    ## LOAD THEM ONE BY ONE
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

# Unfreeze ONLY query tokens + Qformer (as in your rough script intent)
qformer.model.query_tokens.requires_grad = True
for p in qformer.model.Qformer.parameters():
    p.requires_grad = True 

for p in regressor.parameters():
    p.requires_grad = True

##### ----------------- ####
#####  6) LOSSES + OPTIM 
##### ----------------- ####
reg_criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    [p for p in qformer.model.parameters() if p.requires_grad] + list(regressor.parameters()),
    lr=1e-4
)

#### TRAIN ONE ITER FUNCTION ####
def train_one_epoch(dataloader):
    qformer.train()
    regressor.train()
   
    total_loss = 0
    for batch in tqdm(dataloader, total=len(dataloader), desc="Training on A20K"):
        images = batch["images"].to(device, non_blocking=True)
        prompts = batch["prompts"]
        gt_scores = batch["gt_scores"].to(device)
        names = batch["image_names"]
        descs = batch["descs"]

        optimizer.zero_grad()
        mm_mean_embeds = qformer(images, prompts, descs)
        pred = regressor(mm_mean_embeds).squeeze(-1)  # (B,)
        reg_loss = reg_criterion(pred, gt_scores)

        loss = reg_loss
        loss.backward()
        optimizer.step()

        total_loss+=float(loss.detach().item())

    return total_loss/len(dataloader)

#### EVALUATE FUNCTION ####
@torch.no_grad()
def evaluate_and_save_df(dataloader, output_csv_path=None, desc_tag="Evaluating"):
    """
    Runs inference, computes SRCC, and returns a pandas DataFrame with per-sample records:
      image_name, prompt, gen_answer, gt_score, pred_score

    If output_csv_path is provided, saves the DataFrame to CSV.
    """
    qformer.eval()
    regressor.eval()

    rows = []
    pred_list = []
    gt_list = []

    for batch in tqdm(dataloader, total=len(dataloader), desc=desc_tag):
        images = batch["images"].to(device, non_blocking=True)
        image_names = batch["image_names"]
        prompts = batch["prompts"]
        descs = batch["descs"]
        gt_scores = batch["gt_scores"].to(device, non_blocking=True)  # (B,)

        mm_mean_embeds = qformer(images, prompts, descs)
        pred = regressor(mm_mean_embeds).squeeze(-1)  # (B,)

        # accumulate for SRCC
        pred_list.append(pred.detach().float().cpu())
        gt_list.append(gt_scores.detach().float().cpu())

        # store per-sample rows for later analysis
        pred_cpu = pred.detach().float().cpu().numpy()
        gt_cpu   = gt_scores.detach().float().cpu().numpy()

        for i in range(len(image_names)):
            rows.append({
                "image_name": image_names[i],
                "prompt": prompts[i],
                "gen_answer": descs[i],
                "gt_score": float(gt_cpu[i]),
                "pred_score": float(pred_cpu[i]),
            })

    pred_scores = torch.cat(pred_list, dim=0).numpy()
    gt_scores   = torch.cat(gt_list, dim=0).numpy()

    srcc = spearmanr_numpy(pred_scores, gt_scores)

    df = pd.DataFrame(rows)

    if output_csv_path is not None:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        print(f"[Saved] Per-sample results CSV: {output_csv_path}")

    return srcc, df

#### MAIN PIPELINE ####
# read the train csv  
fraction = 0.1
train_df = pd.read_csv("/home/rajivs/anatapmitra/anatap_data/generated_descriptions_PT1/A3K_new/a3k_train_full_gen_responses_PT1_normalized.csv")
train_labeld_df = train_df.sample(frac=fraction, random_state=SEED).reset_index(drop=True)
 
# now get the val and test csv
val_csv = "/home/rajivs/anatapmitra/anatap_data/generated_descriptions_PT1/A3K_new/a3k_val_full_gen_responses_PT1_normalized.csv"
val_df = pd.read_csv("/home/rajivs/anatapmitra/anatap_data/generated_descriptions_PT1/A3K_new/a3k_val_full_gen_responses_PT1_normalized.csv")
test_csv = "/home/rajivs/anatapmitra/anatap_data/generated_descriptions_PT1/A3K_new/a3k_test_full_gen_responses_PT1_normalized.csv"
test_df = pd.read_csv("/home/rajivs/anatapmitra/anatap_data/generated_descriptions_PT1/A3K_new/a3k_test_full_gen_responses_PT1_normalized.csv")

# now get the image_root 
image_root = "/home/rajivs/anatapmitra/anatap_data/anatap_1/agiqa-3k/images"

test_out_csv = "/home/rajivs/anatapmitra/anatap_data/Analysis_EXP/check_a3k_qf_bs_ver_ft_234.csv"
# get data and dataloaders
train_data = QFormerQualityDataset(df=train_labeld_df, img_root=image_root,)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=collate_fn)
# print lengths  
print("length of train_data is:", len(train_data))
val_dataset = QFormerQualityDataset(csv_path=val_csv, img_root=image_root)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
print("length of val_data is:", len(val_dataset))
test_dataset = QFormerQualityDataset(csv_path=test_csv, img_root=image_root)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
print("length of test_data is:", len(test_dataset))
#### Now get the main training and evaluation ####
no_epochs = 15

best_val_srcc, best_test_srcc = -1.0 ,-1.0
for epoch in range(no_epochs):
    # train one epoch 
    train_loss = train_one_epoch(train_loader)
    print(f"Train loss at epoch {epoch+1} is {train_loss}")
    # validate 
    val_srcc,_ = evaluate_and_save_df(val_loader, output_csv_path=None, desc_tag="Evaluating on a20k validation")
    print(f"Validation srcc at epoch {epoch+1} is {val_srcc}")
    if val_srcc > best_val_srcc:
        best_val_srcc = val_srcc
        # then test it 
        test_srcc,_ = evaluate_and_save_df(test_loader, output_csv_path=test_out_csv, desc_tag="Evaluating on a20k test")
        print("Test evaluated")
        print(f"Test srcc at epoch {epoch+1} is {test_srcc}")
        if test_srcc > best_test_srcc:
            best_test_srcc = test_srcc

     

print(f"best test srcc found to be {best_test_srcc}")
print("QL check ver 2 on A20K with bs 16")