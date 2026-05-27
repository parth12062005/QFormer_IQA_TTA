"""
Parameter unfreezing strategies for TTA.

Controls which parameters of the Q-Former are updated during test-time adaptation.
"""

import torch.nn as nn


def get_layernorm_params(qformer_module):
    """Collect all LayerNorm weight/bias parameters from the Q-Former."""
    ln_params = []
    for module in qformer_module.modules():
        if isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                ln_params.append(module.weight)
            if module.bias is not None:
                ln_params.append(module.bias)
    return ln_params


def get_tta_params(qformer_wrapper, strategy: str):
    """
    Return the list of parameters to unfreeze for TTA.

    Args:
        qformer_wrapper: QformerWrapper instance (has .model with .query_tokens and .Qformer)
        strategy: one of "layernorm", "query", "both"

    Returns:
        List of nn.Parameter objects to optimize.
    """
    strategy = strategy.lower()

    if strategy == "none":
        return []
    elif strategy == "layernorm":
        return get_layernorm_params(qformer_wrapper.model.Qformer)
    elif strategy == "query":
        return [qformer_wrapper.model.query_tokens]
    elif strategy == "both":
        return [qformer_wrapper.model.query_tokens] + get_layernorm_params(qformer_wrapper.model.Qformer)
    else:
        raise ValueError(f"Unknown unfreeze strategy: '{strategy}'. Choose from: none, layernorm, query, both")


def freeze_all_except(qformer_wrapper, params_to_update):
    """
    Freeze all parameters in the Q-Former wrapper except those in params_to_update.
    Enables grad for the selected params.
    """
    update_ids = {id(p) for p in params_to_update}

    for p in qformer_wrapper.parameters():
        p.requires_grad = id(p) in update_ids


def print_param_summary(qformer_wrapper, strategy: str):
    """Print a summary of which parameters will be updated during TTA."""
    params = get_tta_params(qformer_wrapper, strategy)
    total = sum(p.numel() for p in params)

    print(f"\n[TTA INFO] Unfreeze strategy: {strategy}")

    if strategy == "none":
        print("  - No parameters unfrozen (frozen baseline with loss computation)")
        print(f"  Total params updated per TTA step: 0\n")
        return

    if strategy in ("query", "both"):
        qt = qformer_wrapper.model.query_tokens
        print(f"  - query_tokens: {list(qt.shape)} → {qt.numel()} params")

    if strategy in ("layernorm", "both"):
        ln_params = get_layernorm_params(qformer_wrapper.model.Qformer)
        ln_count = sum(1 for m in qformer_wrapper.model.Qformer.modules() if isinstance(m, nn.LayerNorm))
        ln_total = sum(p.numel() for p in ln_params)
        print(f"  - Q-Former LayerNorms ({ln_count} layers): {ln_total} params")

    print(f"  Total params updated per TTA step: {total}\n")
