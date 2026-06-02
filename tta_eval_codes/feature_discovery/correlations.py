"""
Correlation and mutual information utilities.
Uses only numpy and scipy — no sklearn.
"""

import numpy as np
from scipy import stats


def pearson_corr(x, y):
    """Pearson correlation coefficient."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return np.nan
    r, _ = stats.pearsonr(x, y)
    return float(r)


def spearman_corr(x, y):
    """Spearman rank correlation coefficient."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return np.nan
    r, _ = stats.spearmanr(x, y)
    return float(r)


def mutual_information(x, y, bins=30):
    """Estimate mutual information using binned histogram method (no sklearn)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 10:
        return np.nan

    # Joint histogram
    hist_xy, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    # Marginals
    hist_x = np.sum(hist_xy, axis=1)
    hist_y = np.sum(hist_xy, axis=0)

    # Normalize to probabilities
    n = float(np.sum(hist_xy))
    if n == 0:
        return 0.0
    p_xy = hist_xy / n
    p_x = hist_x / n
    p_y = hist_y / n

    # MI = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
    mi = 0.0
    for i in range(len(p_x)):
        for j in range(len(p_y)):
            if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
    return float(mi)


def compute_all_correlations(feature_values, mos_values):
    """
    Compute Pearson, Spearman, and MI for a single feature vs MOS.
    
    Args:
        feature_values: 1D array of feature values
        mos_values: 1D array of MOS scores
        
    Returns:
        dict with 'pearson', 'spearman', 'mutual_info' keys
    """
    return {
        "pearson": pearson_corr(feature_values, mos_values),
        "spearman": spearman_corr(feature_values, mos_values),
        "mutual_info": mutual_information(feature_values, mos_values),
    }


def split_stability(feature_values, mos_values, n_splits=5, seed=42):
    """
    Evaluate correlation stability across random 50/50 splits.
    
    Returns:
        dict with 'pearson_mean', 'pearson_std', 'spearman_mean', 'spearman_std'
    """
    rng = np.random.RandomState(seed)
    n = len(feature_values)
    pearsons, spearmans = [], []

    for _ in range(n_splits):
        idx = rng.permutation(n)
        half = n // 2
        for subset in [idx[:half], idx[half:]]:
            fv = feature_values[subset]
            mv = mos_values[subset]
            pearsons.append(pearson_corr(fv, mv))
            spearmans.append(spearman_corr(fv, mv))

    pearsons = [p for p in pearsons if not np.isnan(p)]
    spearmans = [s for s in spearmans if not np.isnan(s)]

    return {
        "pearson_mean": float(np.mean(pearsons)) if pearsons else np.nan,
        "pearson_std": float(np.std(pearsons)) if pearsons else np.nan,
        "spearman_mean": float(np.mean(spearmans)) if spearmans else np.nan,
        "spearman_std": float(np.std(spearmans)) if spearmans else np.nan,
    }
