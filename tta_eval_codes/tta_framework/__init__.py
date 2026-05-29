"""TTA Framework for IQA Evaluation."""
from .losses import GCLoss, RankLoss, FAGCLoss, AdaptiveRankLoss, EMAConsistencyLoss, LOSS_REGISTRY
from .param_strategy import get_tta_params, get_layernorm_params
from .augmentations import (
    create_blur_weak, create_blur_strong,
    create_noise_weak, create_noise_strong,
    create_comp_weak, create_comp_strong,
    create_all_augmentations,
    create_weak_augmentation, create_strong_augmentation,
)
from .engine import TTAEngine
from .strategies import (
    init_proj_head_identity, init_proj_head_orthogonal,
    init_proj_head_jl_gaussian, init_proj_head_jl_sparse,
    init_proj_head_jl_rademacher,
    PROJ_INIT_REGISTRY, EMAProjectionHead,
)

__all__ = [
    "GCLoss", "RankLoss", "FAGCLoss", "AdaptiveRankLoss",
    "EMAConsistencyLoss", "LOSS_REGISTRY",
    "get_tta_params", "get_layernorm_params",
    "create_blur_weak", "create_blur_strong",
    "create_noise_weak", "create_noise_strong",
    "create_comp_weak", "create_comp_strong",
    "create_all_augmentations",
    "create_weak_augmentation", "create_strong_augmentation",
    "TTAEngine",
    "init_proj_head_identity", "init_proj_head_orthogonal",
    "init_proj_head_jl_gaussian", "init_proj_head_jl_sparse",
    "init_proj_head_jl_rademacher",
    "PROJ_INIT_REGISTRY", "EMAProjectionHead",
]
