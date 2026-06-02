# ============================================================
# Cross-database inference: EvalMI-pretrained BASELINE Q-Former -> HPDv2 test set
# ============================================================
# Baseline checkpoint source model:
#   image + prompt -> BLIP-2 Q-Former MM query mean -> regressor -> quality score
#
# HPDv2 cross-database evaluation:
#   score_1 = baseline_model(Image1, Prompt)
#   score_2 = baseline_model(Image2, Prompt)
#   pair_logit = score_1 - score_2
#   prediction = Image1 preferred iff pair_logit > 0
#
# This script:
#   - loads the EvalMI baseline Q-Former checkpoint only;
#   - performs inference only on HPDv2 test pairs;
#   - freezes all parameters and verifies zero trainable parameters;
#   - reports pairwise accuracy and a diagnostic BCE loss;
#   - does NOT fine-tune and does NOT save any CSV file.
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

# Checkpoint saved by baseline Q-Former EvalMI pretraining:
# keys expected: "qformer.Qformer", "query_tokens", "regressor"
EVALMI_BASELINE_CKPT_PATH = (
"/home/rajivs/anatapmitra/anatap_data/Qformer_experiments/new_pretraining/evalmi_baseline_qf_ver2.pth"
)

# HPDv2 test paths.
HPD_TEST_CSV = (
    "/home/rajivs/anatapmitra/anatap_data/"
    "generated_descriptions_PT1/Hpd_new/Hpd_test_full_gen_responses_PT1_full.csv"
)

HPD_TEST_IMG_ROOT = (
    "/home/rajivs/anatapmitra/anatap_data/"
    "hpd_data/test_images/test"
)

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
PIN_MEMORY = True

IMG1_COL = "Image1"
IMG2_COL = "Image2"
PROMPT_COL = "Prompt"
LABEL1_COL = "Label1"
LABEL2_COL = "Label2"


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
# 3) METRIC
# ============================================================

def pairwise_accuracy_from_pairlogit(
    pair_logit: np.ndarray,
    pair_target: np.ndarray,
) -> float:
    """
    pair_logit > 0: Image1 predicted better than Image2.
    pair_target = 1: Label1 > Label2.
    """
    pair_logit = np.asarray(pair_logit)
    pair_target = np.asarray(pair_target).astype(np.float32)

    if pair_logit.size == 0:
        raise ValueError("Cannot compute pairwise accuracy on an empty prediction array.")

    pair_pred = (pair_logit > 0).astype(np.float32)
    return float((pair_pred == pair_target).mean())


# ============================================================
# 4) HPDv2 DATASET + COLLATE
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


class HPDPairwiseTestDataset(Dataset):
    """
    HPDv2 test dataset for baseline cross-database inference.

    Only image pairs, the common prompt, and preference labels are loaded.
    Generated text responses are intentionally unused by the Q-Former baseline.
    """

    def __init__(
        self,
        csv_path: str,
        img_root: str,
        image_tf=None,
        verify_images: bool = True,
        verbose: bool = True,
    ):
        self.df = pd.read_csv(csv_path)
        self.df.columns = self.df.columns.str.strip()
        self.img_root = img_root
        self.image_tf = image_tf if image_tf is not None else qformer_transform

        required_cols = [
            IMG1_COL,
            IMG2_COL,
            PROMPT_COL,
            LABEL1_COL,
            LABEL2_COL,
        ]
        missing_cols = [column for column in required_cols if column not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required HPDv2 columns: {missing_cols}")

        if verify_images:
            valid_indices = []
            dropped = 0
            original_size = len(self.df)

            for idx in range(original_size):
                row = self.df.iloc[idx]
                path1 = os.path.join(self.img_root, str(row[IMG1_COL]))
                path2 = os.path.join(self.img_root, str(row[IMG2_COL]))

                try:
                    if not os.path.isfile(path1) or not os.path.isfile(path2):
                        dropped += 1
                        continue

                    with Image.open(path1) as image1:
                        image1.verify()
                    with Image.open(path2) as image2:
                        image2.verify()

                    valid_indices.append(idx)
                except Exception:
                    dropped += 1

            self.df = self.df.iloc[valid_indices].reset_index(drop=True)

            if verbose:
                print(
                    f"[HPDPairwiseTestDataset] input={original_size} | "
                    f"valid={len(self.df)} | dropped={dropped}"
                )

        if len(self.df) == 0:
            raise RuntimeError("No valid HPDv2 test pairs found.")

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, image_name: str) -> torch.Tensor:
        image_path = os.path.join(self.img_root, image_name)
        image = Image.open(image_path).convert("RGB")
        return self.image_tf(image)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        return {
            "image1": self._load_image(str(row[IMG1_COL])),
            "image2": self._load_image(str(row[IMG2_COL])),
            "prompt": str(row[PROMPT_COL]),
            "label1": torch.tensor(float(row[LABEL1_COL]), dtype=torch.float32),
            "label2": torch.tensor(float(row[LABEL2_COL]), dtype=torch.float32),
        }


def collate_fn(batch):
    return {
        "images1": torch.stack([sample["image1"] for sample in batch], dim=0),
        "images2": torch.stack([sample["image2"] for sample in batch], dim=0),
        "prompts": [sample["prompt"] for sample in batch],
        "label1": torch.stack([sample["label1"] for sample in batch], dim=0),
        "label2": torch.stack([sample["label2"] for sample in batch], dim=0),
    }


# ============================================================
# 5) BASELINE Q-FORMER ARCHITECTURE
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
    EvalMI baseline inference pathway:
        image + prompt
            -> BLIP-2 visual encoder
            -> Q-Former multimodal query outputs
            -> mean query feature [B, 768]
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
            image_embeds = self.model.ln_vision(
                self.model.visual_encoder(images)
            )
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
        return mm_query_embeds.mean(dim=1).float()


# ============================================================
# 6) LOAD BASELINE EVALMI CHECKPOINT + FREEZE
# ============================================================

def load_evalmi_baseline_checkpoint(
    checkpoint_path: str,
    qformer: BaselineQFormerWrapper,
    regressor: Regressor,
) -> None:
    """
    Loads only the components saved by baseline EvalMI pretraining:
        qformer.Qformer, query_tokens, regressor
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"EvalMI baseline checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    required_keys = ["qformer.Qformer", "query_tokens", "regressor"]
    missing_keys = [key for key in required_keys if key not in checkpoint]
    if missing_keys:
        raise KeyError(f"Checkpoint missing required baseline keys: {missing_keys}")

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

    print(f"Loaded EvalMI baseline checkpoint from: {checkpoint_path}")


def freeze_and_verify_inference_only(
    qformer: BaselineQFormerWrapper,
    regressor: Regressor,
) -> None:
    """
    Explicitly prevents any fine-tuning and verifies the inference-only setup.
    """
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
    assert num_trainable == 0, "Some parameters are unexpectedly trainable."


# ============================================================
# 7) CROSS-DATABASE HPDv2 EVALUATION
# ============================================================

@torch.inference_mode()
def evaluate_hpd_pairwise(
    qformer: BaselineQFormerWrapper,
    regressor: Regressor,
    dataloader: DataLoader,
    device: torch.device,
):
    qformer.eval()
    regressor.eval()

    # Diagnostic loss: the baseline was trained as scalar regression on EvalMI;
    # pairwise accuracy is the primary HPDv2 cross-database metric.
    bce_criterion = nn.BCEWithLogitsLoss(reduction="sum")

    total_bce_loss = 0.0
    num_pairs = 0
    pair_logits_list = []
    pair_targets_list = []

    for batch in tqdm(dataloader, total=len(dataloader), desc="Evaluate HPDv2 test"):
        images1 = batch["images1"].to(device, non_blocking=True)
        images2 = batch["images2"].to(device, non_blocking=True)
        prompts = batch["prompts"]
        label1 = batch["label1"].to(device, non_blocking=True)
        label2 = batch["label2"].to(device, non_blocking=True)

        batch_size = images1.size(0)

        images_2b = torch.cat([images1, images2], dim=0)
        prompts_2b = prompts + prompts

        mm_features = qformer(images_2b, prompts_2b)
        predicted_quality = regressor(mm_features).squeeze(-1)

        predicted_quality1, predicted_quality2 = torch.split(
            predicted_quality,
            batch_size,
            dim=0,
        )

        pair_logits = predicted_quality1 - predicted_quality2
        pair_targets = (label1 > label2).float()

        total_bce_loss += float(bce_criterion(pair_logits, pair_targets).item())
        num_pairs += batch_size

        pair_logits_list.append(pair_logits.cpu().numpy())
        pair_targets_list.append(pair_targets.cpu().numpy())

    if num_pairs == 0:
        raise RuntimeError("No HPDv2 test pairs were evaluated.")

    pair_logits_all = np.concatenate(pair_logits_list, axis=0)
    pair_targets_all = np.concatenate(pair_targets_list, axis=0)

    pairwise_accuracy = pairwise_accuracy_from_pairlogit(
        pair_logits_all,
        pair_targets_all,
    )
    average_bce_loss = total_bce_loss / num_pairs

    return average_bce_loss, pairwise_accuracy, num_pairs


# ============================================================
# 8) MAIN
# ============================================================

def main() -> None:
    print("CROSS-DATABASE EVALUATION: EvalMI baseline ver2 Q-Former -> HPDv2 test set")
    print("Model pathway: image + prompt -> MM query features -> baseline regressor")
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

    test_dataset = HPDPairwiseTestDataset(
        csv_path=HPD_TEST_CSV,
        img_root=HPD_TEST_IMG_ROOT,
        image_tf=qformer_transform,
        verify_images=True,
        verbose=True,
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

    test_bce_loss, test_pair_acc, num_test_pairs = evaluate_hpd_pairwise(
        qformer=qformer,
        regressor=regressor,
        dataloader=test_loader,
        device=device,
    )

    print("\n============================================================")
    print("CROSS-DATABASE EVALUATION RESULT")
    print("============================================================")
    print(f"Number of evaluated HPDv2 test pairs : {num_test_pairs}")
    print(f"HPDv2 diagnostic pairwise BCE loss    : {test_bce_loss:.6f}")
    print(f"HPDv2 pairwise accuracy               : {test_pair_acc:.6f}")
    print("============================================================")


if __name__ == "__main__":
    main()