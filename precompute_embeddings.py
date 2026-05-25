"""
Precompute frozen ViT image embeddings for the EvalMi-50K dataset.
Uses both GPUs via DataParallel to speed up extraction.
Saves one .pt file per image (float16) under an embeddings/ directory.

Usage:
    python3 precompute_embeddings.py
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from lavis.models import load_model_and_preprocess


##### ------------- ####
#####  1) CONFIG
##### ------------- ####
IMG_ROOT = "../EvalMi-50K/AIGI2025"
CSV_PATHS = [
    "../EvalMi-50K/evalmi_train.csv",
    "../EvalMi-50K/evalmi_val.csv",
    "../EvalMi-50K/evalmi_test.csv",
]
EMBED_OUT_DIR = "../EvalMi-50K/embeddings"
BATCH_SIZE = 100  # larger batch since we're only doing inference
NUM_WORKERS = 4


##### ------------- ####
#####  2) DATASET
##### ------------- ####
class ImageOnlyDataset(Dataset):
    """Loads images and returns them with their names (no labels needed)."""
    def __init__(self, image_names, img_root):
        self.image_names = image_names
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
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        path = os.path.join(self.img_root, name)
        image = Image.open(path).convert("RGB")
        image = self.image_tf(image)
        return image, name


##### ------------- ####
#####  3) ViT WRAPPER (DataParallel-safe)
##### ------------- ####
class ViTExtractor(nn.Module):
    """
    Thin wrapper around BLIP2's frozen visual_encoder + ln_vision.
    This module ONLY does the ViT forward pass, so DataParallel works fine.
    """
    def __init__(self, visual_encoder, ln_vision):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.ln_vision = ln_vision

    @torch.no_grad()
    def forward(self, images):
        with torch.cuda.amp.autocast():
            embeds = self.ln_vision(self.visual_encoder(images))
        return embeds.float()


##### ------------- ####
#####  4) MAIN
##### ------------- ####
def main():
    # Collect all unique image names across all splits
    all_image_names = set()
    for csv_path in CSV_PATHS:
        df = pd.read_csv(csv_path)
        all_image_names.update(df["image_name"].tolist())
    all_image_names = sorted(all_image_names)

    # Filter out already-computed embeddings
    remaining = []
    for name in all_image_names:
        out_path = os.path.join(EMBED_OUT_DIR, name.replace(".png", ".npz").replace(".jpg", ".npz"))
        if not os.path.exists(out_path):
            remaining.append(name)

    print(f"Total unique images: {len(all_image_names)}")
    print(f"Already computed:    {len(all_image_names) - len(remaining)}")
    print(f"Remaining:           {len(remaining)}")

    if len(remaining) == 0:
        print("All embeddings already precomputed. Nothing to do!")
        return

    # Load BLIP2 model on CPU first, then extract ViT parts
    print("Loading BLIP2 model...")
    device = torch.device("cuda:0")
    model, _, _ = load_model_and_preprocess(
        name="blip2_feature_extractor",
        model_type="pretrain",
        is_eval=True,
        device=device,
    )

    # Extract the frozen ViT encoder and wrap it
    vit = ViTExtractor(model.visual_encoder, model.ln_vision).eval()

    # Use DataParallel across both GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"Using DataParallel across {num_gpus} GPUs")
        vit = nn.DataParallel(vit)
    vit = vit.to(device)

    # Free the rest of the BLIP2 model from GPU memory
    del model
    torch.cuda.empty_cache()

    # Create dataloader
    dataset = ImageOnlyDataset(remaining, IMG_ROOT)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # Extract and save embeddings
    print("Extracting embeddings...")
    with torch.no_grad():
        for images, names in tqdm(dataloader, desc="Precomputing ViT embeddings"):
            images = images.to(device, non_blocking=True)
            embeds = vit(images)  # (B, num_patches, embed_dim)

            # Save each embedding individually as compressed float16 numpy
            for i, name in enumerate(names):
                out_path = os.path.join(
                    EMBED_OUT_DIR,
                    name.replace(".png", ".npz").replace(".jpg", ".npz"),
                )
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                np.savez_compressed(out_path, embed=embeds[i].half().cpu().numpy())

    print(f"Done! Embeddings saved to: {EMBED_OUT_DIR}")

    # Print shape info for reference
    sample_path = os.path.join(
        EMBED_OUT_DIR,
        remaining[0].replace(".png", ".npz").replace(".jpg", ".npz"),
    )
    sample = np.load(sample_path)["embed"]
    print(f"Embedding shape per image: {sample.shape} (dtype: {sample.dtype})")
    print(f"File size: {os.path.getsize(sample_path) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
