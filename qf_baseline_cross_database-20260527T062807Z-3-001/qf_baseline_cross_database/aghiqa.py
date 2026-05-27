# ============================================================
# Cross-database inference: EvalMI-pretrained baseline Q-Former -> AIGIQA-20K test set
# ============================================================
# Baseline model:
#   image + prompt -> BLIP-2 Q-Former MM query mean -> regressor -> predicted quality
#
# This script:
#   1) Loads the baseline EvalMI checkpoint:
#        - qformer.Qformer
#        - query_tokens
#        - regressor
#   2) Freezes all model parameters.
#   3) Runs inference only on the A20K test set.
#   4) Reports cross-database SROCC.
#   5) Does not train, fine-tune, or save prediction CSVs.
# ============================================================

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


# ============================================================
# 1) CONFIG
# ============================================================

SEED = 1234
# Baseline Q-Former checkpoint trained on EvalMI.
EVALMI_BASELINE_CKPT_PATH = (
    "/home/rajivs/anatapmitra/anatap_data/Qformer_experiments/new_pretraining/evalmi_baseline_qf_ver2.pth"
)

# AIGIQA-20K target/test database.
A20K_TEST_CSV = (
    "/storage/users/rajivs/anatapmitra/AGHI-QA/train_test_split/aghiqa_test_full_gen_responses_PT1.csv"
)

A20K_IMG_ROOT = (
    "/storage/users/rajivs/anatapmitra/AGHI-QA/images/images"
)

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
PIN_MEMORY = True

IMG_COL = "image_name"
PROMPT_COL = "prompt"
GT_COL = "gt_score"


# ============================================================
# 2) REPRODUCIBILITY
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 3) METRIC: SROCC
# ============================================================

def rankdata_numpy(a: np.ndarray) -> np.ndarray:
    """
    NumPy equivalent of scipy.stats.rankdata(method="average").
    Correctly assigns average ranks for tied values.
    """
    a = np.asarray(a)
    sorter = np.argsort(a)
    inverse = np.empty_like(sorter)
    inverse[sorter] = np.arange(len(a))

    a_sorted = a[sorter]
    new_value = np.concatenate(([True], a_sorted[1:] != a_sorted[:-1]))
    dense_rank = np.cumsum(new_value)

    counts = np.bincount(dense_rank)
    cumulative = np.cumsum(counts)

    ranks = (cumulative[dense_rank] + cumulative[dense_rank - 1] + 1) / 2.0
    return ranks[inverse]


def spearmanr_numpy(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape. Got {x.shape} and {y.shape}.")

    rx = rankdata_numpy(x)
    ry = rankdata_numpy(y)

    rx = rx - rx.mean()
    ry = ry - ry.mean()

    denominator = np.sqrt(np.sum(rx ** 2) * np.sum(ry ** 2))
    if denominator == 0:
        return float("nan")

    return float(np.sum(rx * ry) / denominator)


# ============================================================
# 4) A20K TEST DATASET
# ============================================================

qformer_transform = transforms.Compose([
    transforms.Resize(
        (IMG_SIZE, IMG_SIZE),
        interpolation=transforms.InterpolationMode.BICUBIC,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    ),
])


class A20KTestDataset(Dataset):
    """
    A20K test dataset for baseline Q-Former cross-database inference.

    Only image, prompt, and ground-truth score are needed because the
    baseline EvalMI model never uses generated descriptions.
    """

    def __init__(self, csv_path: str, img_root: str, image_tf=None):
        self.df = pd.read_csv(csv_path)
        self.df.columns = self.df.columns.str.strip()
        self.img_root = img_root
        self.image_tf = image_tf if image_tf is not None else qformer_transform

        required_cols = [IMG_COL, PROMPT_COL, GT_COL]
        missing_cols = [column for column in required_cols if column not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in A20K test CSV: {missing_cols}")

        if len(self.df) == 0:
            raise RuntimeError("A20K test CSV contains no samples.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_name = str(row[IMG_COL])
        image_path = os.path.join(self.img_root, image_name)

        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        image = self.image_tf(image)

        return {
            "image": image,
            "prompt": str(row[PROMPT_COL]),
            "gt_score": torch.tensor(float(row[GT_COL]), dtype=torch.float32),
        }


def collate_fn(batch):
    return {
        "images": torch.stack([item["image"] for item in batch], dim=0),
        "prompts": [item["prompt"] for item in batch],
        "gt_scores": torch.stack([item["gt_score"] for item in batch], dim=0),
    }


# ============================================================
# 5) BASELINE Q-FORMER MODEL
# ============================================================

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


class BaselineQFormerWrapper(nn.Module):
    """
    Baseline MM inference pathway:
        image + prompt
            -> frozen BLIP-2 visual encoder
            -> Q-Former multimodal pass
            -> mean of Q-Former query outputs [B, 768]
    """

    def __init__(self, device: torch.device):
        super().__init__()

        model, _, _ = load_model_and_preprocess(
            name="blip2_feature_extractor",
            model_type="pretrain",
            is_eval=True,
            device=device,
        )

        self.model = model.to(device)
        self.device = device

    def forward(self, images: torch.Tensor, prompts: list[str]) -> torch.Tensor:
        batch_size = images.size(0)
        images = images.to(self.device, non_blocking=True)

        with self.model.maybe_autocast():
            image_embeds = self.model.ln_vision(self.model.visual_encoder(images))
        image_embeds = image_embeds.float()

        image_atts = torch.ones(
            image_embeds.size()[:-1],
            dtype=torch.long,
            device=self.device,
        )

        query_tokens = self.model.query_tokens.expand(batch_size, -1, -1)

        text_prompt = self.model.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        query_atts = torch.ones(
            query_tokens.size()[:-1],
            dtype=torch.long,
            device=self.device,
        )

        mm_attention_mask = torch.cat(
            [query_atts, text_prompt.attention_mask],
            dim=1,
        )

        mm_out = self.model.Qformer.bert(
            text_prompt.input_ids,
            query_embeds=query_tokens,
            attention_mask=mm_attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        mm_query_embeds = mm_out.last_hidden_state[:, :query_tokens.size(1), :]
        mm_mean_embeds = mm_query_embeds.mean(dim=1)

        return mm_mean_embeds.float()


# ============================================================
# 6) CHECKPOINT LOADING AND FREEZING
# ============================================================

def load_evalmi_baseline_checkpoint(
    checkpoint_path: str,
    qformer: BaselineQFormerWrapper,
    regressor: Regressor,
) -> None:
    """
    Loads exactly the keys saved by the EvalMI baseline pretraining code:
        qformer.Qformer, query_tokens, regressor
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Baseline EvalMI checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    required_keys = ["qformer.Qformer", "query_tokens", "regressor"]
    missing_keys = [key for key in required_keys if key not in checkpoint]
    if missing_keys:
        raise KeyError(f"Checkpoint is missing required baseline keys: {missing_keys}")

    qformer.model.Qformer.load_state_dict(
        checkpoint["qformer.Qformer"],
        strict=True,
    )

    with torch.no_grad():
        qformer.model.query_tokens.copy_(
            checkpoint["query_tokens"].to(
                qformer.model.query_tokens.device,
                dtype=qformer.model.query_tokens.dtype,
            )
        )

    regressor.load_state_dict(
        checkpoint["regressor"],
        strict=True,
    )

    print(f"Loaded baseline EvalMI checkpoint from: {checkpoint_path}")


def freeze_and_verify_inference_only(
    qformer: BaselineQFormerWrapper,
    regressor: Regressor,
) -> None:
    for parameter in qformer.parameters():
        parameter.requires_grad = False

    for parameter in regressor.parameters():
        parameter.requires_grad = False

    qformer.eval()
    regressor.eval()

    num_trainable = (
        sum(parameter.numel() for parameter in qformer.parameters() if parameter.requires_grad)
        + sum(parameter.numel() for parameter in regressor.parameters() if parameter.requires_grad)
    )

    print(f"Number of trainable parameters during inference: {num_trainable}")
    assert num_trainable == 0, "Inference setup error: some parameters remain trainable."


# ============================================================
# 7) CROSS-DATABASE EVALUATION
# ============================================================

@torch.inference_mode()
def evaluate_a20k(
    qformer: BaselineQFormerWrapper,
    regressor: Regressor,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    qformer.eval()
    regressor.eval()

    prediction_list = []
    gt_list = []

    for batch in tqdm(dataloader, total=len(dataloader), desc="Evaluate A20K test"):
        images = batch["images"].to(device, non_blocking=True)
        prompts = batch["prompts"]
        gt_scores = batch["gt_scores"].to(device, non_blocking=True)

        mm_features = qformer(images, prompts)
        predictions = regressor(mm_features).squeeze(-1)

        prediction_list.append(predictions.float().cpu())
        gt_list.append(gt_scores.float().cpu())

    predicted_scores = torch.cat(prediction_list, dim=0).numpy()
    ground_truth_scores = torch.cat(gt_list, dim=0).numpy()

    return spearmanr_numpy(predicted_scores, ground_truth_scores)


# ============================================================
# 8) MAIN
# ============================================================

def main() -> None:
    print("CROSS-DATABASE EVALUATION: EvalMI baseline ver 2 Q-Former -> AGHIQA test set")
    print("Model pathway: image + prompt -> MM query features -> regressor")
    print("Fine-tuning: disabled")
    print("Prediction CSV saving: disabled")

    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    qformer = BaselineQFormerWrapper(device=device).to(device)
    regressor = Regressor(input_dim=768, hidden_dim=256, output_dim=1).to(device)

    load_evalmi_baseline_checkpoint(
        checkpoint_path=EVALMI_BASELINE_CKPT_PATH,
        qformer=qformer,
        regressor=regressor,
    )

    freeze_and_verify_inference_only(
        qformer=qformer,
        regressor=regressor,
    )

    test_dataset = A20KTestDataset(
        csv_path=A20K_TEST_CSV,
        img_root=A20K_IMG_ROOT,
        image_tf=qformer_transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_fn,
        drop_last=False,
    )

    test_srcc = evaluate_a20k(
        qformer=qformer,
        regressor=regressor,
        dataloader=test_loader,
        device=device,
    )

    print("\n============================================================")
    print("CROSS-DATABASE EVALUATION RESULT")
    print("============================================================")
    print(f"Number of evaluated AGHIQA test images : {len(test_dataset)}")
    print(f"AGHIQA test SROCC                      : {test_srcc:.6f}")
    print("============================================================")


if __name__ == "__main__":
    main()