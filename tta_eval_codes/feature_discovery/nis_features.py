"""
Natural Image Statistics (NIS) features for image quality assessment.
Uses only numpy and scipy.
"""
import numpy as np
from scipy import ndimage, stats as sp_stats


def _to_gray(img):
    if img.ndim == 3:
        return np.dot(img[..., :3].astype(np.float64), [0.2989, 0.5870, 0.1140])
    return img.astype(np.float64)


def _local_mean_std(gray, kernel_size=7):
    mu = ndimage.uniform_filter(gray, size=kernel_size)
    sq = ndimage.uniform_filter(gray ** 2, size=kernel_size)
    sigma = np.sqrt(np.maximum(sq - mu**2, 0))
    return mu, sigma


def _mscn_coefficients(gray, kernel_size=7):
    mu, sigma = _local_mean_std(gray, kernel_size)
    return (gray - mu) / (sigma + 1.0)


def _fit_ggd_shape(x):
    x = x.ravel()
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return 2.0
    m2 = np.mean(x**2)
    m4 = np.mean(x**4)
    if m2 < 1e-12:
        return 2.0
    kappa = m4 / (m2**2 + 1e-12)
    if kappa <= 1.5:
        return 0.5
    elif kappa >= 6.0:
        return 4.0
    else:
        return max(0.1, min(10.0, 3.0 / (kappa - 1.0 + 1e-8)))


def _entropy_hist(data, bins=256):
    hist, _ = np.histogram(data.ravel(), bins=bins, density=True)
    hist = hist / (hist.sum() + 1e-12)
    return float(-np.sum(hist[hist > 0] * np.log2(hist[hist > 0])))


def compute_nis_features(img):
    gray = _to_gray(img)
    features = {}
    mscn = _mscn_coefficients(gray)
    mscn_flat = mscn.ravel()
    features["mscn_mean"] = float(np.mean(mscn))
    features["mscn_std"] = float(np.std(mscn))
    features["mscn_kurtosis"] = float(sp_stats.kurtosis(mscn_flat, fisher=True))
    features["mscn_skewness"] = float(sp_stats.skew(mscn_flat))
    features["mscn_ggd_shape"] = float(_fit_ggd_shape(mscn))

    gx = ndimage.sobel(gray, axis=1)
    gy = ndimage.sobel(gray, axis=0)
    grad_mag = np.sqrt(gx**2 + gy**2)
    features["gradient_magnitude_mean"] = float(np.mean(grad_mag))
    features["gradient_magnitude_std"] = float(np.std(grad_mag))
    features["gradient_entropy"] = float(_entropy_hist(grad_mag, bins=128))

    laplacian = ndimage.laplace(gray)
    features["laplacian_var"] = float(np.var(laplacian))
    features["laplacian_energy"] = float(np.mean(laplacian**2))

    grad_threshold = np.mean(grad_mag) + np.std(grad_mag)
    features["edge_density"] = float(np.sum(grad_mag > grad_threshold) / gray.size)

    mu, sigma = _local_mean_std(gray, kernel_size=15)
    lc = sigma / (mu + 1e-12)
    features["local_contrast_mean"] = float(np.mean(lc))
    features["local_contrast_std"] = float(np.std(lc))

    block_size = 32
    h, w = gray.shape
    ents = []
    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            ents.append(_entropy_hist(gray[y:y+block_size, x:x+block_size], 64))
    features["local_entropy_mean"] = float(np.mean(ents)) if ents else 0.0
    features["local_entropy_std"] = float(np.std(ents)) if ents else 0.0

    return features


NIS_FEATURE_META = {
    "mscn_mean": {"description": "Mean of MSCN coefficients", "differentiable": True, "batch_computable": True, "family": "nis"},
    "mscn_std": {"description": "Std of MSCN coefficients", "differentiable": True, "batch_computable": True, "family": "nis"},
    "mscn_kurtosis": {"description": "Kurtosis of MSCN (BRISQUE-style)", "differentiable": False, "batch_computable": True, "family": "nis"},
    "mscn_skewness": {"description": "Skewness of MSCN distribution", "differentiable": False, "batch_computable": True, "family": "nis"},
    "mscn_ggd_shape": {"description": "GGD shape parameter of MSCN", "differentiable": False, "batch_computable": True, "family": "nis"},
    "gradient_magnitude_mean": {"description": "Mean Sobel gradient magnitude", "differentiable": True, "batch_computable": True, "family": "nis"},
    "gradient_magnitude_std": {"description": "Std of gradient magnitude", "differentiable": True, "batch_computable": True, "family": "nis"},
    "gradient_entropy": {"description": "Entropy of gradient histogram", "differentiable": False, "batch_computable": True, "family": "nis"},
    "laplacian_var": {"description": "Variance of Laplacian (sharpness)", "differentiable": True, "batch_computable": True, "family": "nis"},
    "laplacian_energy": {"description": "Mean squared Laplacian response", "differentiable": True, "batch_computable": True, "family": "nis"},
    "edge_density": {"description": "Fraction of edge pixels", "differentiable": False, "batch_computable": True, "family": "nis"},
    "local_contrast_mean": {"description": "Mean local_std/local_mean", "differentiable": True, "batch_computable": True, "family": "nis"},
    "local_contrast_std": {"description": "Std of local contrast", "differentiable": True, "batch_computable": True, "family": "nis"},
    "local_entropy_mean": {"description": "Mean blockwise pixel entropy", "differentiable": False, "batch_computable": True, "family": "nis"},
    "local_entropy_std": {"description": "Std of blockwise pixel entropy", "differentiable": False, "batch_computable": True, "family": "nis"},
}
