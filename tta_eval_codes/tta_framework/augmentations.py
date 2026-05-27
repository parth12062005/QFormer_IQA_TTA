"""
Image augmentations for rank-based TTA losses.

These augmentations are applied to CLIP-preprocessed image tensors
(already resized to 224x224, normalized) before re-extraction through ViT.
"""

import torch
from torchvision import transforms


def create_weak_augmentation(images: torch.Tensor) -> torch.Tensor:
    """
    Apply weak Gaussian blur (σ ∈ [5, 10]).
    Represents mild distortion — features should be close to original.
    """
    blur = transforms.GaussianBlur(kernel_size=5, sigma=(5.0, 10.0))
    return blur(images)


def create_strong_augmentation(images: torch.Tensor) -> torch.Tensor:
    """
    Apply strong Gaussian blur (σ ∈ [30, 40]).
    Represents heavy distortion — features should be far from original.
    """
    blur = transforms.GaussianBlur(kernel_size=5, sigma=(30.0, 40.0))
    return blur(images)
