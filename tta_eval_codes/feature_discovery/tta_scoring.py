"""
TTA Suitability Scorer.
"""


def score_tta_suitability(feature_name, abs_spearman, split_std, feature_meta):
    """
    Score a feature's suitability for Test-Time Adaptation on a scale of 0 to 10.
    
    Criteria (max 10 points):
      1. MOS Correlation (|Spearman|): up to 3 points
         > 0.6 = 3, > 0.4 = 2, > 0.2 = 1, else 0
      2. Computable without labels (Self-Supervised): 2 points
         All features in this framework are computed from images/embeddings without MOS,
         so this is effectively always 2 points.
      3. Differentiable: 3 points
         If True in meta, 3 points. (Needed for gradient descent).
      4. Batch-computable: 1 point
         If True in meta, 1 point.
      5. Scale-stable (split consistency): 1 point
         If split_std < 0.05, 1 point.
    """
    meta = feature_meta.get(feature_name, {})
    
    score = 0
    
    # 1. Correlation
    if abs_spearman >= 0.6:
        score += 3
    elif abs_spearman >= 0.4:
        score += 2
    elif abs_spearman >= 0.2:
        score += 1
        
    # 2. Computable without labels
    score += 2
    
    # 3. Differentiable
    if meta.get("differentiable", False):
        score += 3
        
    # 4. Batch computable
    if meta.get("batch_computable", True):
        score += 1
        
    # 5. Stability
    if split_std < 0.05:
        score += 1
        
    return score


def generate_tta_recommendation(feature_name, meta):
    """Generate a text recommendation for how to use this feature for TTA."""
    if not meta.get("differentiable", False):
        return "Not directly differentiable. Use as a pseudo-label generator, sample weighting, or clustering criterion (like FAGC/LLM-TTA)."
        
    family = meta.get("family", "misc")
    if family == "frequency":
        return "Use as a frequency-domain regularizer (e.g., L2 loss to match source domain spectral flatness)."
    elif family == "nis":
        return "Use as an auxiliary naturalness objective (e.g., minimize distance to expected MSCN distribution)."
    elif family == "deep":
        return "Use as a representation constraint (e.g., entropy minimization or contrastive alignment)."
    elif family == "patch":
        return "Use as an internal consistency loss (maximize similarity of augmented patches)."
    else:
        return "Formulate as a direct auxiliary loss term minimized via gradient descent."
