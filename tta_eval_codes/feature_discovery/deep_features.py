"""
Deep features from precomputed ViT embeddings + CLIP/DINO models.
"""
import numpy as np


def compute_deep_features_from_embed(embed):
    """
    Compute deep features from precomputed ViT embeddings.
    
    Args:
        embed: numpy array (257, 1408) — CLS + 256 patch tokens
    
    Returns:
        dict[str, float]
    """
    features = {}
    embed = embed.astype(np.float64)
    
    cls_token = embed[0]  # (1408,)
    patch_tokens = embed[1:]  # (256, 1408)

    # CLS token norm
    features["embed_cls_norm"] = float(np.linalg.norm(cls_token))

    # Patch token norms
    patch_norms = np.linalg.norm(patch_tokens, axis=1)
    features["embed_patch_norm_mean"] = float(np.mean(patch_norms))
    features["embed_patch_norm_std"] = float(np.std(patch_norms))

    # CLS entropy (softmax then entropy)
    cls_sm = np.exp(cls_token - np.max(cls_token))
    cls_sm = cls_sm / (cls_sm.sum() + 1e-12)
    features["embed_entropy"] = float(-np.sum(cls_sm[cls_sm > 0] * np.log(cls_sm[cls_sm > 0])))

    # Patch pairwise cosine similarity (subsample for speed)
    n_patches = patch_tokens.shape[0]
    norms = patch_norms[:, None] + 1e-12
    patch_normed = patch_tokens / norms
    
    # Use random subset if too many patches
    if n_patches > 64:
        idx = np.random.RandomState(42).choice(n_patches, 64, replace=False)
        patch_sub = patch_normed[idx]
    else:
        patch_sub = patch_normed
    
    sim_matrix = patch_sub @ patch_sub.T
    n_sub = len(patch_sub)
    triu_idx = np.triu_indices(n_sub, k=1)
    pairwise_sims = sim_matrix[triu_idx]
    features["embed_patch_similarity_mean"] = float(np.mean(pairwise_sims))
    features["embed_patch_similarity_std"] = float(np.std(pairwise_sims))

    # SVD of patch tokens (subsample columns for speed)
    try:
        if patch_tokens.shape[1] > 512:
            col_idx = np.random.RandomState(42).choice(patch_tokens.shape[1], 512, replace=False)
            pt_sub = patch_tokens[:, col_idx]
        else:
            pt_sub = patch_tokens
        
        U, S, Vt = np.linalg.svd(pt_sub, full_matrices=False)
        S_total = np.sum(S) + 1e-12
        
        features["embed_singular_value_ratio"] = float(S[0] / S_total)
        
        # Effective rank = exp(entropy of normalized singular values)
        S_norm = S / S_total
        S_norm = S_norm[S_norm > 0]
        eff_rank_entropy = -np.sum(S_norm * np.log(S_norm))
        features["embed_effective_rank"] = float(np.exp(eff_rank_entropy))
        
        # Top-5 PCA energy
        top5_energy = np.sum(S[:5]**2)
        total_sq = np.sum(S**2) + 1e-12
        features["embed_top5_pca_energy"] = float(top5_energy / total_sq)
    except Exception:
        features["embed_singular_value_ratio"] = 0.0
        features["embed_effective_rank"] = 0.0
        features["embed_top5_pca_energy"] = 0.0

    # Covariance trace and norms
    cov = np.cov(patch_tokens.T) if patch_tokens.shape[0] > 1 else np.zeros((1, 1))
    features["embed_covariance_trace"] = float(np.trace(cov))
    features["embed_nuclear_norm"] = float(np.sum(S)) if 'S' in dir() else 0.0
    features["embed_frobenius_norm"] = float(np.linalg.norm(patch_tokens, 'fro'))

    return features


def compute_clip_features(img_tensor, clip_model, clip_preprocess, device):
    """
    Compute CLIP embedding features from a raw image tensor.
    
    Args:
        img_tensor: preprocessed image tensor (1, 3, 224, 224)
        clip_model: loaded CLIP model
        clip_preprocess: CLIP preprocessing transform
        device: torch device
    
    Returns:
        dict[str, float]
    """
    import torch
    features = {}
    
    with torch.no_grad():
        img_feat = clip_model.encode_image(img_tensor.to(device))
        feat = img_feat.float().cpu().numpy().flatten()
    
    features["clip_embed_norm"] = float(np.linalg.norm(feat))
    
    # Entropy of softmax
    sm = np.exp(feat - np.max(feat))
    sm = sm / (sm.sum() + 1e-12)
    features["clip_embed_entropy"] = float(-np.sum(sm[sm > 0] * np.log(sm[sm > 0])))
    
    # Sparsity (L1/L2)
    l1 = np.sum(np.abs(feat))
    l2 = np.sqrt(np.sum(feat**2)) + 1e-12
    features["clip_embed_sparsity"] = float(l1 / (l2 * np.sqrt(len(feat))))
    
    # Stats
    features["clip_embed_mean"] = float(np.mean(feat))
    features["clip_embed_std"] = float(np.std(feat))
    features["clip_embed_kurtosis"] = float(
        np.mean(((feat - np.mean(feat)) / (np.std(feat) + 1e-12))**4) - 3
    )
    
    return features


def compute_dino_features(img_tensor, dino_model, device):
    """
    Compute DINO embedding features.
    
    Args:
        img_tensor: preprocessed image tensor (1, 3, 224, 224)
        dino_model: loaded DINO model  
        device: torch device
    
    Returns:
        dict[str, float]
    """
    import torch
    features = {}
    
    with torch.no_grad():
        feat = dino_model(img_tensor.to(device))
        feat = feat.float().cpu().numpy().flatten()
    
    features["dino_embed_norm"] = float(np.linalg.norm(feat))
    
    sm = np.exp(feat - np.max(feat))
    sm = sm / (sm.sum() + 1e-12)
    features["dino_embed_entropy"] = float(-np.sum(sm[sm > 0] * np.log(sm[sm > 0])))
    
    l1 = np.sum(np.abs(feat))
    l2 = np.sqrt(np.sum(feat**2)) + 1e-12
    features["dino_embed_sparsity"] = float(l1 / (l2 * np.sqrt(len(feat))))
    
    features["dino_embed_mean"] = float(np.mean(feat))
    features["dino_embed_std"] = float(np.std(feat))
    features["dino_embed_kurtosis"] = float(
        np.mean(((feat - np.mean(feat)) / (np.std(feat) + 1e-12))**4) - 3
    )
    
    return features


DEEP_FEATURE_META = {
    "embed_cls_norm": {"description": "L2 norm of ViT CLS token", "differentiable": True, "batch_computable": True, "family": "deep"},
    "embed_patch_norm_mean": {"description": "Mean L2 norm of ViT patch tokens", "differentiable": True, "batch_computable": True, "family": "deep"},
    "embed_patch_norm_std": {"description": "Std of ViT patch L2 norms", "differentiable": True, "batch_computable": True, "family": "deep"},
    "embed_entropy": {"description": "Entropy of softmax(CLS token)", "differentiable": True, "batch_computable": True, "family": "deep"},
    "embed_patch_similarity_mean": {"description": "Mean pairwise cosine sim of patches", "differentiable": True, "batch_computable": True, "family": "deep"},
    "embed_patch_similarity_std": {"description": "Std of pairwise patch cosine sim", "differentiable": True, "batch_computable": True, "family": "deep"},
    "embed_singular_value_ratio": {"description": "s1/sum(s) of patch embedding SVD", "differentiable": False, "batch_computable": True, "family": "deep"},
    "embed_effective_rank": {"description": "exp(entropy of normalized singular values)", "differentiable": False, "batch_computable": True, "family": "deep"},
    "embed_top5_pca_energy": {"description": "Fraction of variance in top 5 PCs", "differentiable": False, "batch_computable": True, "family": "deep"},
    "embed_covariance_trace": {"description": "Trace of patch covariance matrix", "differentiable": False, "batch_computable": True, "family": "deep"},
    "embed_nuclear_norm": {"description": "Nuclear norm (sum of singular values)", "differentiable": True, "batch_computable": True, "family": "deep"},
    "embed_frobenius_norm": {"description": "Frobenius norm of patch embeddings", "differentiable": True, "batch_computable": True, "family": "deep"},
    "clip_embed_norm": {"description": "L2 norm of CLIP image embedding", "differentiable": True, "batch_computable": True, "family": "deep"},
    "clip_embed_entropy": {"description": "Entropy of softmax(CLIP embedding)", "differentiable": True, "batch_computable": True, "family": "deep"},
    "clip_embed_sparsity": {"description": "L1/(L2*sqrt(N)) of CLIP embedding", "differentiable": True, "batch_computable": True, "family": "deep"},
    "clip_embed_mean": {"description": "Mean of CLIP embedding", "differentiable": True, "batch_computable": True, "family": "deep"},
    "clip_embed_std": {"description": "Std of CLIP embedding", "differentiable": True, "batch_computable": True, "family": "deep"},
    "clip_embed_kurtosis": {"description": "Kurtosis of CLIP embedding", "differentiable": False, "batch_computable": True, "family": "deep"},
    "dino_embed_norm": {"description": "L2 norm of DINO embedding", "differentiable": True, "batch_computable": True, "family": "deep"},
    "dino_embed_entropy": {"description": "Entropy of softmax(DINO embedding)", "differentiable": True, "batch_computable": True, "family": "deep"},
    "dino_embed_sparsity": {"description": "L1/(L2*sqrt(N)) of DINO embedding", "differentiable": True, "batch_computable": True, "family": "deep"},
    "dino_embed_mean": {"description": "Mean of DINO embedding", "differentiable": True, "batch_computable": True, "family": "deep"},
    "dino_embed_std": {"description": "Std of DINO embedding", "differentiable": True, "batch_computable": True, "family": "deep"},
    "dino_embed_kurtosis": {"description": "Kurtosis of DINO embedding", "differentiable": False, "batch_computable": True, "family": "deep"},
}
