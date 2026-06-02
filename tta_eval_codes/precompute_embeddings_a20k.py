"""
Precompute frozen ViT image embeddings for the AGIQA-20K dataset.
Saves one .npz file per image (float16) under an embeddings/a20k/ directory.

Usage:
    python3 precompute_embeddings_a20k.py
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
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Image root contains train/, val/, test/ subdirs with flat image files
IMG_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "../../agiqa-20k/AIGCQA-30K-Image"))

# Split CSVs from the important split files folder
_SPLIT_DIR = os.path.join(
    _SCRIPT_DIR,
    "../important split files-20260527T062853Z-3-001",
    "important split files",
    "A20K_new",
)
CSV_PATHS = [
    os.path.join(_SPLIT_DIR, "A20k_train_full_PT1_normalized.csv"),
    os.path.join(_SPLIT_DIR, "A20k_val_full_PT1_normalized.csv"),
    os.path.join(_SPLIT_DIR, "A20k_test_full_PT1_normalized.csv"),
]

EMBED_OUT_DIR = "/media/parth/021f75bf-bae8-49ef-86a5-28ca19171835/parth/dataset/embeddings/a20k"
BATCH_SIZE = 500
NUM_WORKERS = 8


def _build_image_lookup(img_root):
    """
    Build a mapping: filename -> full path.
    Images are spread across train/, val/, test/ subdirs,
    so we scan all subdirs to find each image.
    """
    lookup = {}
    for subdir in os.listdir(img_root):
        subdir_path = os.path.join(img_root, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for fname in os.listdir(subdir_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                lookup[fname] = os.path.join(subdir_path, fname)
    return lookup


##### ------------- ####
#####  2) DATASET
##### ------------- ####
class ImageOnlyDataset(Dataset):
    """Loads images and returns them with their names (no labels needed)."""
    def __init__(self, image_names, image_lookup):
        self.image_names = image_names
        self.image_lookup = image_lookup
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
        path = self.image_lookup[name]
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
    # Build image lookup from disk
    print(f"Scanning image directory: {IMG_ROOT}")
    image_lookup = _build_image_lookup(IMG_ROOT)
    print(f"Found {len(image_lookup)} images on disk")

    # Collect all unique image names across all splits
    all_image_names = set()
    for csv_path in CSV_PATHS:
        print(f"Reading: {csv_path}")
        df = pd.read_csv(csv_path)
        all_image_names.update(df["image_name"].tolist())
    all_image_names = sorted(all_image_names)

    # Check which images are missing from disk
    missing_on_disk = [n for n in all_image_names if n not in image_lookup]
    if missing_on_disk:
        print(f"WARNING: {len(missing_on_disk)} images in CSVs not found on disk!")
        for m in missing_on_disk[:5]:
            print(f"  - {m}")
        if len(missing_on_disk) > 5:
            print(f"  ... and {len(missing_on_disk) - 5} more")

    # Filter out already-computed embeddings
    os.makedirs(EMBED_OUT_DIR, exist_ok=True)
    remaining = []
    for name in all_image_names:
        if name not in image_lookup:
            continue  # skip images not on disk
        out_path = os.path.join(EMBED_OUT_DIR, name.replace(".png", ".npz").replace(".jpg", ".npz"))
        if not os.path.exists(out_path):
            remaining.append(name)

    print(f"\nTotal unique images in CSVs: {len(all_image_names)}")
    print(f"Available on disk:          {len(all_image_names) - len(missing_on_disk)}")
    print(f"Already computed:           {len(all_image_names) - len(missing_on_disk) - len(remaining)}")
    print(f"Remaining to compute:       {len(remaining)}")

    if len(remaining) == 0:
        print("All embeddings already precomputed. Nothing to do!")
        return

    # Load BLIP2 model
    print("\nLoading BLIP2 model...")
    device = torch.device("cuda:0")
    model, _, _ = load_model_and_preprocess(
        name="blip2_feature_extractor",
        model_type="pretrain",
        is_eval=True,
        device=device,
    )

    # Extract the frozen ViT encoder and wrap it
    vit = ViTExtractor(model.visual_encoder, model.ln_vision).eval()

    # Use DataParallel across multiple GPUs if available
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"Using DataParallel across {num_gpus} GPUs")
        vit = nn.DataParallel(vit)
    vit = vit.to(device)

    # Free the rest of the BLIP2 model from GPU memory
    del model
    torch.cuda.empty_cache()

    # Create dataloader
    dataset = ImageOnlyDataset(remaining, image_lookup)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # Extract and save embeddings
    print("Extracting embeddings...")
    with torch.no_grad():
        for images, names in tqdm(dataloader, desc="Precomputing ViT embeddings (A20K)"):
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

    print(f"\nDone! Embeddings saved to: {EMBED_OUT_DIR}")

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
