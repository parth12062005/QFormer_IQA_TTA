"""TTA Framework for IQA Evaluation."""
from .losses import GCLoss, RankLoss, FAGCLoss, AdaptiveRankLoss, LOSS_REGISTRY
from .param_strategy import get_tta_params, get_layernorm_params
from .augmentations import (
    create_blur_weak, create_blur_strong,
    create_noise_weak, create_noise_strong,
    create_comp_weak, create_comp_strong,
    create_all_augmentations,
    create_weak_augmentation, create_strong_augmentation,
)
from .engine import TTAEngine

__all__ = [
    "GCLoss", "RankLoss", "FAGCLoss", "AdaptiveRankLoss", "LOSS_REGISTRY",
    "get_tta_params", "get_layernorm_params",
    "create_blur_weak", "create_blur_strong",
    "create_noise_weak", "create_noise_strong",
    "create_comp_weak", "create_comp_strong",
    "create_all_augmentations",
    "create_weak_augmentation", "create_strong_augmentation",
    "TTAEngine",
]
