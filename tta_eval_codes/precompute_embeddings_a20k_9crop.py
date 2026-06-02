"""
Precompute frozen ViT image embeddings for AGIQA-20K using 9-crop (3x3 grid).

Each 1024x1024 image is split into a 3x3 grid of ~341x341 non-overlapping patches,
each resized to 224x224. All 9 crops are passed through the ViT encoder.
Saves one .npz file per image containing all 9 crop embeddings.

Usage:
    python precompute_embeddings_a20k_9crop.py
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

IMG_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "../../agiqa-20k/AIGCQA-30K-Image"))

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

EMBED_OUT_DIR = "/media/parth/021f75bf-bae8-49ef-86a5-28ca19171835/parth/dataset/embeddings/a20k_9crop"
BATCH_SIZE = 250   # 250 images × 9 crops = 2250 ViT fwd passes per batch
NUM_WORKERS = 8


def _build_image_lookup(img_root):
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
#####  2) 9-CROP DATASET
##### ------------- ####
CLIP_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    ),
])


def get_9crops(pil_img):
    """Split a PIL image into a 3x3 grid of non-overlapping crops."""
    w, h = pil_img.size
    cw, ch = w // 3, h // 3
    crops = []
    for row in range(3):
        for col in range(3):
            x0, y0 = col * cw, row * ch
            x1 = x0 + cw if col < 2 else w
            y1 = y0 + ch if row < 2 else h
            crop = pil_img.crop((x0, y0, x1, y1))
            crops.append(CLIP_TRANSFORM(crop))
    return torch.stack(crops)  # (9, 3, 224, 224)


class NineCropImageDataset(Dataset):
    """Loads images and returns 9 crops with their names."""
    def __init__(self, image_names, image_lookup):
        self.image_names = image_names
        self.image_lookup = image_lookup

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        path = self.image_lookup[name]
        image = Image.open(path).convert("RGB")
        crops = get_9crops(image)  # (9, 3, 224, 224)
        return crops, name


##### ------------- ####
#####  3) ViT WRAPPER
##### ------------- ####
class ViTExtractor(nn.Module):
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
    print(f"Scanning image directory: {IMG_ROOT}")
    image_lookup = _build_image_lookup(IMG_ROOT)
    print(f"Found {len(image_lookup)} images on disk")

    all_image_names = set()
    for csv_path in CSV_PATHS:
        print(f"Reading: {csv_path}")
        df = pd.read_csv(csv_path)
        all_image_names.update(df["image_name"].tolist())
    all_image_names = sorted(all_image_names)

    missing_on_disk = [n for n in all_image_names if n not in image_lookup]
    if missing_on_disk:
        print(f"WARNING: {len(missing_on_disk)} images in CSVs not found on disk!")

    os.makedirs(EMBED_OUT_DIR, exist_ok=True)
    remaining = []
    for name in all_image_names:
        if name not in image_lookup:
            continue
        out_path = os.path.join(EMBED_OUT_DIR, name.replace(".png", ".npz").replace(".jpg", ".npz"))
        if not os.path.exists(out_path):
            remaining.append(name)

    print(f"\nTotal unique images in CSVs: {len(all_image_names)}")
    print(f"Available on disk:          {len(all_image_names) - len(missing_on_disk)}")
    print(f"Already computed:           {len(all_image_names) - len(missing_on_disk) - len(remaining)}")
    print(f"Remaining to compute:       {len(remaining)}")

    if len(remaining) == 0:
        print("All 9-crop embeddings already precomputed. Nothing to do!")
        return

    print("\nLoading BLIP2 model...")
    device = torch.device("cuda:0")
    model, _, _ = load_model_and_preprocess(
        name="blip2_feature_extractor",
        model_type="pretrain",
        is_eval=True,
        device=device,
    )

    vit = ViTExtractor(model.visual_encoder, model.ln_vision).eval()
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"Using DataParallel across {num_gpus} GPUs")
        vit = nn.DataParallel(vit)
    vit = vit.to(device)

    del model
    torch.cuda.empty_cache()

    dataset = NineCropImageDataset(remaining, image_lookup)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    print(f"Extracting 9-crop embeddings for {len(remaining)} images...")
    with torch.no_grad():
        for crops_batch, names in tqdm(dataloader, desc="Precomputing 9-crop ViT embeddings"):
            # crops_batch: (B, 9, 3, 224, 224)
            B = crops_batch.size(0)
            flat_crops = crops_batch.view(B * 9, 3, 224, 224).to(device, non_blocking=True)
            flat_embeds = vit(flat_crops)  # (B*9, 257, 1408)
            flat_embeds = flat_embeds.view(B, 9, *flat_embeds.shape[1:])  # (B, 9, 257, 1408)

            for i, name in enumerate(names):
                out_path = os.path.join(
                    EMBED_OUT_DIR,
                    name.replace(".png", ".npz").replace(".jpg", ".npz"),
                )
                # Save all 9 crop embeddings together as float16
                np.savez_compressed(out_path, embed=flat_embeds[i].half().cpu().numpy())

    print(f"\nDone! 9-crop embeddings saved to: {EMBED_OUT_DIR}")

    sample_path = os.path.join(
        EMBED_OUT_DIR,
        remaining[0].replace(".png", ".npz").replace(".jpg", ".npz"),
    )
    sample = np.load(sample_path)["embed"]
    print(f"Embedding shape per image: {sample.shape} (dtype: {sample.dtype})")
    print(f"  Expected: (9, 257, 1408)")
    print(f"File size: {os.path.getsize(sample_path) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
