"""
Patch consistency features for image quality assessment.
"""
import numpy as np
from scipy import ndimage


def _to_gray(img):
    if img.ndim == 3:
        return np.dot(img[..., :3].astype(np.float64), [0.2989, 0.5870, 0.1140])
    return img.astype(np.float64)


def _entropy_hist(data, bins=64):
    hist, _ = np.histogram(data.ravel(), bins=bins, density=True)
    hist = hist / (hist.sum() + 1e-12)
    return float(-np.sum(hist[hist > 0] * np.log2(hist[hist > 0])))


def compute_patch_features(img, patch_size=64):
    """
    Compute patch-consistency features.
    
    Args:
        img: numpy array HWC uint8
        patch_size: size of square patches
    
    Returns:
        dict[str, float]
    """
    gray = _to_gray(img)
    h, w = gray.shape
    features = {}
    ps = min(patch_size, h // 2, w // 2)
    if ps < 16:
        ps = 16

    # Extract patches
    patch_means = []
    patch_stds = []
    patch_entropies = []
    patch_spectral_energies = []
    patch_spectral_slopes = []

    for y in range(0, h - ps + 1, ps):
        for x in range(0, w - ps + 1, ps):
            patch = gray[y:y+ps, x:x+ps]
            patch_means.append(np.mean(patch))
            patch_stds.append(np.std(patch))
            patch_entropies.append(_entropy_hist(patch, 64))
            
            # Patch FFT
            p_fft = np.abs(np.fft.fftshift(np.fft.fft2(patch)))
            p_power = p_fft ** 2
            patch_spectral_energies.append(np.sum(p_power))
            
            # Patch spectral slope
            radial = []
            pcx, pcy = ps // 2, ps // 2
            for r in range(1, pcx):
                Y, X = np.ogrid[:ps, :ps]
                ring = (np.sqrt((X - pcy)**2 + (Y - pcx)**2) >= r - 0.5) & \
                       (np.sqrt((X - pcy)**2 + (Y - pcx)**2) < r + 0.5)
                if ring.any():
                    radial.append(np.mean(p_power[ring]))
            if len(radial) > 3:
                log_f = np.log(np.arange(1, len(radial) + 1))
                log_p = np.log(np.array(radial) + 1e-12)
                mask = np.isfinite(log_f) & np.isfinite(log_p)
                if mask.sum() > 2:
                    slope = np.polyfit(log_f[mask], log_p[mask], 1)[0]
                    patch_spectral_slopes.append(slope)

    # Color patch consistency (if color image)
    if img.ndim == 3:
        color_means = []
        for y in range(0, h - ps + 1, ps):
            for x in range(0, w - ps + 1, ps):
                patch = img[y:y+ps, x:x+ps].astype(np.float64)
                color_means.append(np.mean(patch, axis=(0, 1)))
        if len(color_means) > 1:
            color_means = np.array(color_means)
            features["patch_color_consistency"] = float(np.mean(np.std(color_means, axis=0)))
        else:
            features["patch_color_consistency"] = 0.0
    else:
        features["patch_color_consistency"] = 0.0

    features["patch_pixel_var"] = float(np.var(patch_means)) if patch_means else 0.0
    features["patch_entropy_var"] = float(np.var(patch_entropies)) if patch_entropies else 0.0
    features["patch_fft_var"] = float(np.var(patch_spectral_energies)) if patch_spectral_energies else 0.0

    # Texture inconsistency: std of spectral slopes
    if len(patch_spectral_slopes) > 1:
        features["patch_texture_similarity"] = float(1.0 / (np.std(patch_spectral_slopes) + 1e-8))
    else:
        features["patch_texture_similarity"] = 0.0

    # Self-similarity: correlation between adjacent patches
    if len(patch_means) > 2:
        pm = np.array(patch_means)
        autocorr = np.correlate(pm - pm.mean(), pm - pm.mean(), mode='full')
        autocorr = autocorr / (autocorr.max() + 1e-12)
        mid = len(autocorr) // 2
        if mid + 1 < len(autocorr):
            features["ssim_self_similarity"] = float(autocorr[mid + 1])
        else:
            features["ssim_self_similarity"] = 0.0
        
        # Texture repetition: how fast autocorrelation decays
        decay = autocorr[mid:]
        if len(decay) > 3:
            features["texture_repetition_score"] = float(np.mean(np.abs(decay[1:4])))
        else:
            features["texture_repetition_score"] = 0.0
    else:
        features["ssim_self_similarity"] = 0.0
        features["texture_repetition_score"] = 0.0

    # LBP uniformity (simplified)
    lbp_vals = []
    for y in range(1, min(h-1, 200)):
        for x in range(1, min(w-1, 200)):
            center = gray[y, x]
            code = 0
            for k, (dy, dx) in enumerate([(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]):
                if gray[y+dy, x+dx] >= center:
                    code |= (1 << k)
            lbp_vals.append(code)
    if lbp_vals:
        hist, _ = np.histogram(lbp_vals, bins=256, range=(0, 256))
        hist_norm = hist / (hist.sum() + 1e-12)
        features["lbp_uniformity"] = float(-np.sum(hist_norm[hist_norm > 0] * np.log2(hist_norm[hist_norm > 0])))
    else:
        features["lbp_uniformity"] = 0.0

    return features


PATCH_FEATURE_META = {
    "patch_pixel_var": {"description": "Variance of per-patch mean pixels", "differentiable": False, "batch_computable": True, "family": "patch"},
    "patch_color_consistency": {"description": "Std of per-patch mean color", "differentiable": False, "batch_computable": True, "family": "patch"},
    "patch_texture_similarity": {"description": "Inverse std of per-patch spectral slopes", "differentiable": False, "batch_computable": True, "family": "patch"},
    "patch_fft_var": {"description": "Variance of per-patch spectral energy", "differentiable": False, "batch_computable": True, "family": "patch"},
    "ssim_self_similarity": {"description": "Lag-1 autocorrelation of patch means", "differentiable": False, "batch_computable": True, "family": "patch"},
    "texture_repetition_score": {"description": "Mean short-lag autocorrelation strength", "differentiable": False, "batch_computable": True, "family": "patch"},
    "lbp_uniformity": {"description": "Entropy of LBP histogram (texture uniformity)", "differentiable": False, "batch_computable": True, "family": "patch"},
    "patch_entropy_var": {"description": "Variance of per-patch entropy", "differentiable": False, "batch_computable": True, "family": "patch"},
}
