#### Script to pretrain the exact configuration of architecture with EvalMI
#### Here we train the qformer + desc. reglartization architecture, but with 100% labels from EvalMI, no role of description for consistency loss
#### and unlabeled loss
# ------------------------------------------------------------

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


# -----------------------------
# 1) SEED
# -----------------------------
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


# -----------------------------
# 2) METRICS (SRCC)  (EXACTLY AS YOU PROVIDED)
# -----------------------------
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


# -----------------------------
# 3) DATASET + COLLATE (UPDATED CLASS)
# -----------------------------
IMG_COL    = "image_name"
PROMPT_COL = "prompt"
DESC_COL   = "gen_answer"
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



# -----------------------------
# 4) DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


# -----------------------------
# 5) MODELS
# -----------------------------
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


class QformerWrapper(nn.Module):
    """
    Returns:
      text_feats: [B, H_text] (output of model.text_proj on CLS from desc-only pass)
      mm_mean:    [B, 768]    (mean over query tokens from multimodal pass)
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

        # (1) image embeds from frozen vision encoder
        with torch.no_grad():
            with self.model.maybe_autocast():
                image_embeds_frozen = self.model.ln_vision(self.model.visual_encoder(images))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long, device=self.device
            )

        query_tokens = self.model.query_tokens.expand(B, -1, -1)

        # (2) text-only pass on descs -> CLS -> text_proj
        text_desc = self.model.tokenizer(
            descs, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        text_out = self.model.Qformer.bert(
            text_desc.input_ids,
            attention_mask=text_desc.attention_mask,
            return_dict=True,
        )
        text_cls = text_out.last_hidden_state[:, 0, :]          # [B,768]
        text_feats = self.model.text_proj(text_cls)             # [B,H_text]

        # (3) multimodal pass on prompts
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
        mm_mean = mm_query_embeds.mean(dim=1)                                      # [B,768]

        return text_feats.float(), mm_mean.float()


# -----------------------------
# 6) INIT + LOAD CKPT (ONLY QFORMER + QUERY TOKENS)
# -----------------------------
qformer = QformerWrapper(device=device, is_eval=False).to(device)

# BLIP2 text_proj output dimension (typically 256)
TEXT_DIM = qformer.model.text_proj.out_features if hasattr(qformer.model.text_proj, "out_features") else 256

regressor_mm = Regressor(input_dim=768, hidden_dim=256, output_dim=1).to(device)
regressor_text = Regressor(input_dim=TEXT_DIM, hidden_dim=128, output_dim=1).to(device)


# -----------------------------
# 7) FREEZE/UNFREEZE
# -----------------------------
for p in qformer.model.parameters():
    p.requires_grad = False

qformer.model.query_tokens.requires_grad = True
for p in qformer.model.Qformer.parameters():
    p.requires_grad = True

# IMPORTANT: you need text_proj trainable if you want text branch to learn regression
for p in qformer.model.text_proj.parameters():
    p.requires_grad = True

for p in regressor_mm.parameters():
    p.requires_grad = True
for p in regressor_text.parameters():
    p.requires_grad = True


# -----------------------------
# 8) LOSS + OPTIM
# -----------------------------
reg_criterion = nn.MSELoss()

optimizer = torch.optim.AdamW(
    [p for p in qformer.model.parameters() if p.requires_grad]
    + list(regressor_mm.parameters())
    + list(regressor_text.parameters()),
    lr=1e-4,
)

#### Train and evaluate per epoch function ####

##### ----------------- ####
#####  7) TRAIN / EVAL
##### ----------------- ####
def train_one_epoch(dataloader):
    qformer.train()
    regressor_mm.train()
    regressor_text.train()

    total_loss = 0.0
    for batch in tqdm(dataloader, total=len(dataloader), desc="Training qformer"):
        images = batch["images"].to(device)
        prompts = batch["prompts"]
        descs = batch["descs"]
        gt_scores = batch["gt_scores"].to(device)  # (B,)

        optimizer.zero_grad()

        text_feat, mm_mean_embeds = qformer(images, prompts, descs)

        pred_mm = regressor_mm(mm_mean_embeds).squeeze(-1)  # (B,)
        pred_text = regressor_text(text_feat).squeeze(-1) #(B,)
        reg_loss = reg_criterion(pred_mm, gt_scores) + reg_criterion(pred_text, gt_scores)

        loss = reg_loss
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().item())
    

    return total_loss / max(1, len(dataloader))



#### test csv save function
@torch.no_grad()
def evaluate_and_save_df(dataloader, output_csv_path=None, desc_tag="Evaluating"):
    """
    Runs inference, computes SRCC, and returns a pandas DataFrame with per-sample records:
      image_name, prompt, gen_answer, gt_score, pred_score

    If output_csv_path is provided, saves the DataFrame to CSV.
    """
    qformer.eval()
    regressor_mm.eval()
    regressor_text.eval()

    rows = []
    pred_mm_list = []
    pred_text_list = []
    gt_list = []

    for batch in tqdm(dataloader, total=len(dataloader), desc=desc_tag):
        images = batch["images"].to(device, non_blocking=True)
        image_names = batch["image_names"]
        prompts = batch["prompts"]
        descs = batch["descs"]
        gt_scores = batch["gt_scores"].to(device, non_blocking=True)  # (B,)

        text_feat, mm_mean_embeds = qformer(images, prompts, descs)

        pred_mm = regressor_mm(mm_mean_embeds).squeeze(-1)  # (B,)
        pred_text = regressor_text(text_feat).squeeze(-1) #(B,)

        # accumulate for SRCC
        pred_mm_list.append(pred_mm.detach().float().cpu())
        pred_text_list.append(pred_text.detach().float().cpu())
        gt_list.append(gt_scores.detach().float().cpu())

        # store per-sample rows for later analysis
        pred_mm_cpu = pred_mm.detach().float().cpu().numpy()
        pred_text_cpu = pred_text.detach().float().cpu().numpy()
        gt_cpu   = gt_scores.detach().float().cpu().numpy()

        for i in range(len(image_names)):
            rows.append({
                "image_name": image_names[i],
                "prompt": prompts[i],
                "gen_answer": descs[i],
                "gt_score": float(gt_cpu[i]),
                "pred_mm_score": float(pred_mm_cpu[i]),
                "pred_text_score": float(pred_text_cpu[i]),
            })

    ## compute srcc
    pred_mm_scores = torch.cat(pred_mm_list, dim=0).numpy()
    gt_scores   = torch.cat(gt_list, dim=0).numpy()

    srcc = spearmanr_numpy(pred_mm_scores, gt_scores)

    df = pd.DataFrame(rows)

    if output_csv_path is not None:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        print(f"[Saved] Per-sample results CSV: {output_csv_path}")
    

    return srcc, df


##### ----------------- ####
#####  8) MAIN
##### ----------------- ####
def main():
    print("PRETRAINING FOR QFORMER + DESCRIPTION REGULARIZATION (Qen3vl MLLM) ON EVALMI")
    train_dataset = QFormerQualityDataset(
        csv_path="/home/rajivs/anatapmitra/anatap_data/QwenVL3/description_generation_mllm/evalmi_train_qwenvl3_full_gen_responses_pretrained.csv",
        img_root="/home/rajivs/anatapmitra/anatap_data/anatap_1/public_datasets/EvalMi-50K",
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
    )

    val_dataset = QFormerQualityDataset(
        csv_path="/home/rajivs/anatapmitra/anatap_data/generated_desceriptions_MLLM/EvalMI/evalmi_val_full_gen_responses_MLLM.csv",
        img_root="/home/rajivs/anatapmitra/anatap_data/anatap_1/public_datasets/EvalMi-50K",
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
    )

    test_dataset = QFormerQualityDataset(
        csv_path="/home/rajivs/anatapmitra/anatap_data/generated_desceriptions_MLLM/EvalMI/evalmi_test_full_gen_responses_MLLM.csv",
        img_root="/home/rajivs/anatapmitra/anatap_data/anatap_1/public_datasets/EvalMi-50K",
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
    )
    
    train_eval_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    checkpoint_path = "/home/rajivs/anatapmitra/anatap_data/Qformer_experiments/new_pretraining/evalmi_qf_desc_reg_mllm_qwen3vl.pth"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    best_srcc, best_test_srcc = -1.0, -1.0
    train_out_csv = "/home/rajivs/anatapmitra/anatap_data/Qformer_experiments/new_pretraining/evalmi_qf_desc_reg_train_mllm.csv"
    test_out_csv = "/home/rajivs/anatapmitra/anatap_data/Qformer_experiments/new_pretraining/evalmi_qf_desc_reg_test_mllm.csv"

    for epoch in range(15):
        train_loss = train_one_epoch(train_dataloader)
        val_srcc,_= evaluate_and_save_df(val_dataloader, output_csv_path=None, desc_tag="Evaluating on evalmi validation")

        print("epoch--", epoch)
        print("train loss--", train_loss)
        print("val evaluated ,Val srcc--", val_srcc)

        if val_srcc > best_srcc:
            best_srcc = val_srcc
            test_srcc,_= evaluate_and_save_df(test_dataloader, output_csv_path=test_out_csv, desc_tag="Evaluating on evalmi test")
            # train_srcc,_ = evaluate_and_save_df(train_eval_dataloader, output_csv_path=train_out_csv, desc_tag="Evaluating on evalmi train set")
            best_test_srcc = test_srcc
            
            # print("train evaluated, Train srcc--", train_srcc)
            print("test evaluated ,Test srcc--", test_srcc)

            torch.save(
                {
                    "qformer.Qformer": qformer.model.Qformer.state_dict(),
                    "query_tokens": qformer.model.query_tokens.detach().cpu(),
                    "regressor_mm": regressor_mm.state_dict(),
                    "regressor_text": regressor_text.state_dict(),
                    "text_proj": qformer.model.text_proj.state_dict(),
                },
                checkpoint_path,
            )
            print("model saved")

        print("best val srcc--", best_srcc)
        print("best test srcc--", best_test_srcc)


if __name__ == "__main__":
    main()