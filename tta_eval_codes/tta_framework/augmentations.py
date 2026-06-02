"""
Image augmentations for rank-based TTA losses.

Three distortion types at two severity levels each, matching the original
TTA-IQA implementation (Roy et al., ICCV 2023):
  1. Gaussian Blur  (weak σ∈[5,20], strong σ∈[40,60])
  2. Gaussian Noise (weak σ=0.02, strong σ=0.08)
  3. JPEG Compression (weak quality=50, strong quality=10)
"""

import io
import torch
from torchvision import transforms
from PIL import Image
import numpy as np


# ── Blur ───────────────────────────────────────────────────────────────────
def create_blur_weak(images: torch.Tensor) -> torch.Tensor:
    """Weak Gaussian blur — σ ∈ [5, 20]."""
    sigma = 5 + np.random.random() * 15
    blur = transforms.GaussianBlur(kernel_size=5, sigma=sigma)
    return blur(images)


def create_blur_strong(images: torch.Tensor) -> torch.Tensor:
    """Strong Gaussian blur — σ ∈ [40, 60]."""
    sigma = 40 + np.random.random() * 20
    blur = transforms.GaussianBlur(kernel_size=5, sigma=sigma)
    return blur(images)


# ── Gaussian Noise ─────────────────────────────────────────────────────────
def create_noise_weak(images: torch.Tensor) -> torch.Tensor:
    """Weak additive Gaussian noise (σ=0.02)."""
    return images + torch.randn_like(images) * 0.02


def create_noise_strong(images: torch.Tensor) -> torch.Tensor:
    """Strong additive Gaussian noise (σ=0.08)."""
    return images + torch.randn_like(images) * 0.08


# ── JPEG Compression (PIL roundtrip) ──────────────────────────────────────
# CLIP normalization constants for un/re-normalizing
_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
_CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)


def _jpeg_compress_tensor(img_tensor: torch.Tensor, quality: int) -> torch.Tensor:
    """Compress a single (3,H,W) CLIP-normalized tensor via JPEG and re-normalize."""
    # Un-normalize
    img = img_tensor.cpu() * _CLIP_STD + _CLIP_MEAN
    img = img.clamp(0, 1)

    # To PIL
    pil_img = Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))

    # JPEG roundtrip in memory
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    pil_img = Image.open(buf).convert("RGB")

    # Back to tensor + re-normalize
    out = transforms.ToTensor()(pil_img)
    out = (out - _CLIP_MEAN) / _CLIP_STD
    return out


def create_comp_weak(images: torch.Tensor) -> torch.Tensor:
    """Weak JPEG compression (quality=50)."""
    return torch.stack([_jpeg_compress_tensor(images[i], quality=50) for i in range(images.size(0))]).to(images.device)


def create_comp_strong(images: torch.Tensor) -> torch.Tensor:
    """Strong JPEG compression (quality=10)."""
    return torch.stack([_jpeg_compress_tensor(images[i], quality=10) for i in range(images.size(0))]).to(images.device)


# ── Convenience: all 6 augmented versions ─────────────────────────────────
def create_all_augmentations(images: torch.Tensor) -> dict:
    """
    Create all 6 augmented versions (3 distortion types × 2 severities).
    Returns dict mapping augmentation name to tensor.
    """
    return {
        "blur_weak":   create_blur_weak(images),
        "blur_strong": create_blur_strong(images),
        "noise_weak":  create_noise_weak(images),
        "noise_strong":create_noise_strong(images),
        "comp_weak":   create_comp_weak(images),
        "comp_strong": create_comp_strong(images),
    }


# ── Legacy API (backwards-compatible) ─────────────────────────────────────
def create_weak_augmentation(images: torch.Tensor) -> torch.Tensor:
    return create_blur_weak(images)

def create_strong_augmentation(images: torch.Tensor) -> torch.Tensor:
    return create_blur_strong(images)
