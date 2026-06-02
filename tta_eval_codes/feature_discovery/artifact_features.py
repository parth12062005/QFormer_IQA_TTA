"""
Generative Artifact Detection features.
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


def compute_artifact_features(img):
    """
    Compute generative artifact detection features.
    
    Args:
        img: numpy array HWC uint8
    
    Returns:
        dict[str, float]
    """
    gray = _to_gray(img)
    features = {}

    # Gradient and Laplacian
    gx = ndimage.sobel(gray, axis=1)
    gy = ndimage.sobel(gray, axis=0)
    grad_mag = np.sqrt(gx**2 + gy**2)
    lap = ndimage.laplace(gray)
    
    mean_grad = np.mean(grad_mag)
    var_lap = np.var(lap)
    
    # Unnatural sharpness: Laplacian energy relative to gradient energy
    features["unnatural_sharpness"] = float(var_lap / (mean_grad**2 + 1e-12))
    
    # Smoothness ratio: fraction of pixels with very low gradient
    smooth_mask = grad_mag < (0.1 * mean_grad + 1e-12)
    features["smoothness_gradient_ratio"] = float(np.sum(smooth_mask) / gray.size)
    
    # Oversmoothing score: low gradient energy combined with low entropy
    ent = _entropy_hist(gray)
    features["oversmoothing_score"] = float(1.0 / (mean_grad * ent + 1e-12))
    
    # Artificial sharpening: excess HF energy
    fft = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    power = fft**2
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((X - cx)**2 + (Y - cy)**2)
    max_r = min(cx, cy)
    hf_mask = r > max_r * 0.75
    hf_energy = np.sum(power[hf_mask]) / (np.sum(power) + 1e-12)
    features["artificial_sharpening_score"] = float(hf_energy * features["unnatural_sharpness"])

    # Frequency spikes
    radial = []
    for rad in range(1, max_r):
        ring = (r >= rad - 0.5) & (r < rad + 0.5)
        if ring.any():
            radial.append(np.mean(power[ring]))
    if len(radial) > 5:
        med_rad = ndimage.median_filter(radial, size=5)
        spikes = np.sum(radial > 3 * med_rad + 1e-12)
        features["frequency_spike_count"] = float(spikes)
    else:
        features["frequency_spike_count"] = 0.0

    # Diffusion grid artifacts: energy at specific grid-aligned frequencies
    features["diffusion_grid_artifact"] = float(
        np.sum(power[cy, :cx//2]) + np.sum(power[:cy//2, cx]) / (np.sum(power) + 1e-12)
    )

    # Color banding (plateau count in histogram)
    if img.ndim == 3:
        plateaus = 0
        for c in range(3):
            hist, _ = np.histogram(img[..., c], bins=256)
            diff = np.diff(hist)
            plateaus += np.sum(diff == 0)
        features["color_banding_score"] = float(plateaus / 3.0)
    else:
        hist, _ = np.histogram(gray, bins=256)
        features["color_banding_score"] = float(np.sum(np.diff(hist) == 0))

    # Repetition artifact: autocorrelation non-decay
    # We take a central patch to compute autocorrelation
    ps = min(256, h, w)
    patch = gray[cy-ps//2:cy+ps//2, cx-ps//2:cx+ps//2]
    patch = patch - np.mean(patch)
    if patch.size > 0:
        p_fft = np.fft.fft2(patch)
        autocorr = np.fft.ifft2(p_fft * np.conj(p_fft)).real
        autocorr = np.fft.fftshift(autocorr)
        ac_cy, ac_cx = autocorr.shape[0]//2, autocorr.shape[1]//2
        # Check peaks outside the center
        autocorr[ac_cy-5:ac_cy+5, ac_cx-5:ac_cx+5] = 0
        max_side_peak = np.max(autocorr)
        center_peak = np.sum(patch**2)
        features["repetition_artifact_score"] = float(max_side_peak / (center_peak + 1e-12))
    else:
        features["repetition_artifact_score"] = 0.0

    # Hallucination and texture inconsistency are already covered by deep/patch features,
    # but we can add placeholders or simple proxies.
    features["texture_inconsistency"] = float(np.std(grad_mag)) # Simple proxy

    return features


ARTIFACT_FEATURE_META = {
    "oversmoothing_score": {"description": "Inverse of (gradient * entropy)", "differentiable": False, "batch_computable": True, "family": "artifact"},
    "artificial_sharpening_score": {"description": "HF energy * unnatural sharpness", "differentiable": False, "batch_computable": True, "family": "artifact"},
    "texture_inconsistency": {"description": "Std of gradient magnitude", "differentiable": False, "batch_computable": True, "family": "artifact"},
    "frequency_spike_count": {"description": "Outlier peaks in frequency spectrum", "differentiable": False, "batch_computable": True, "family": "artifact"},
    "diffusion_grid_artifact": {"description": "Energy at grid-aligned frequencies", "differentiable": False, "batch_computable": True, "family": "artifact"},
    "color_banding_score": {"description": "Plateau count in color histogram", "differentiable": False, "batch_computable": True, "family": "artifact"},
    "repetition_artifact_score": {"description": "Autocorrelation non-decay score", "differentiable": False, "batch_computable": True, "family": "artifact"},
    "smoothness_gradient_ratio": {"description": "Fraction of pixels with very low gradient", "differentiable": False, "batch_computable": True, "family": "artifact"},
    "unnatural_sharpness": {"description": "Laplacian energy / gradient energy ratio", "differentiable": True, "batch_computable": True, "family": "artifact"},
}
