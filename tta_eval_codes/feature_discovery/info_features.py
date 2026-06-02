"""
Information-theoretic features for image quality assessment.
"""
import io
import numpy as np
from PIL import Image


def _to_gray(img):
    if img.ndim == 3:
        return np.dot(img[..., :3].astype(np.float64), [0.2989, 0.5870, 0.1140])
    return img.astype(np.float64)


def _entropy_hist(data, bins=256):
    hist, _ = np.histogram(data.ravel(), bins=bins, density=True)
    hist = hist / (hist.sum() + 1e-12)
    return float(-np.sum(hist[hist > 0] * np.log2(hist[hist > 0])))


def _compression_ratio(img_pil, fmt, quality=None):
    """Compute compression ratio: compressed_size / raw_size."""
    raw_size = img_pil.size[0] * img_pil.size[1] * 3  # width * height * channels
    buf = io.BytesIO()
    kwargs = {}
    if fmt == "JPEG" and quality:
        kwargs["quality"] = quality
    img_pil.save(buf, format=fmt, **kwargs)
    compressed_size = buf.tell()
    return compressed_size / (raw_size + 1e-12)


def compute_info_features(img):
    """
    Compute information-theoretic features.
    
    Args:
        img: numpy array HWC uint8
    
    Returns:
        dict[str, float]
    """
    gray = _to_gray(img)
    features = {}

    # Global image entropy
    features["image_entropy"] = _entropy_hist(gray, bins=256)

    # Color entropy (joint histogram approximation)
    if img.ndim == 3:
        # Per-channel entropy averaged
        ch_ents = []
        for c in range(3):
            ch_ents.append(_entropy_hist(img[:, :, c], bins=256))
        features["color_entropy"] = float(np.mean(ch_ents))
        
        # Mutual information between R and G channels
        r, g, b = img[:,:,0].ravel().astype(np.float64), img[:,:,1].ravel().astype(np.float64), img[:,:,2].ravel().astype(np.float64)
        hist_rg, _, _ = np.histogram2d(r, g, bins=32)
        hist_r = np.sum(hist_rg, axis=1)
        hist_g = np.sum(hist_rg, axis=0)
        n = hist_rg.sum() + 1e-12
        p_rg = hist_rg / n
        p_r = hist_r / n
        p_g = hist_g / n
        mi = 0.0
        for i in range(32):
            for j in range(32):
                if p_rg[i, j] > 0 and p_r[i] > 0 and p_g[j] > 0:
                    mi += p_rg[i, j] * np.log(p_rg[i, j] / (p_r[i] * p_g[j]))
        features["mutual_info_rgb_channels"] = float(mi)
    else:
        features["color_entropy"] = features["image_entropy"]
        features["mutual_info_rgb_channels"] = 0.0

    # Compression ratios
    img_pil = Image.fromarray(img) if img.ndim == 3 else Image.fromarray(img.astype(np.uint8))
    if img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")
    features["jpeg_compression_ratio"] = float(_compression_ratio(img_pil, "JPEG", quality=75))
    features["png_compression_ratio"] = float(_compression_ratio(img_pil, "PNG"))

    # Spatial complexity (gradient energy / area)
    from scipy import ndimage
    gx = ndimage.sobel(gray, axis=1)
    gy = ndimage.sobel(gray, axis=0)
    features["spatial_complexity"] = float(np.mean(gx**2 + gy**2))

    # Fractal dimension (box-counting, simplified)
    h, w = gray.shape
    threshold = np.median(gray)
    binary = (gray > threshold).astype(np.uint8)
    
    sizes = []
    counts = []
    for s in [4, 8, 16, 32, 64]:
        if s >= min(h, w):
            continue
        count = 0
        for y in range(0, h - s + 1, s):
            for x in range(0, w - s + 1, s):
                if np.any(binary[y:y+s, x:x+s]):
                    count += 1
        if count > 0:
            sizes.append(s)
            counts.append(count)
    
    if len(sizes) >= 2:
        log_s = np.log(1.0 / np.array(sizes))
        log_c = np.log(np.array(counts).astype(np.float64))
        coeffs = np.polyfit(log_s, log_c, 1)
        features["fractal_dimension"] = float(coeffs[0])
    else:
        features["fractal_dimension"] = 2.0

    # Spectral complexity: number of significant frequency components
    fft = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    threshold_spec = np.mean(fft) + 2 * np.std(fft)
    features["spectral_complexity"] = float(np.sum(fft > threshold_spec) / fft.size)

    # Colorfulness (Hasler-Süsstrunk metric)
    if img.ndim == 3:
        R, G, B = img[:,:,0].astype(np.float64), img[:,:,1].astype(np.float64), img[:,:,2].astype(np.float64)
        rg = R - G
        yb = 0.5 * (R + G) - B
        sigma_rgyb = np.sqrt(np.std(rg)**2 + np.std(yb)**2)
        mu_rgyb = np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)
        features["colorfulness"] = float(sigma_rgyb + 0.3 * mu_rgyb)
    else:
        features["colorfulness"] = 0.0

    # Saturation mean
    if img.ndim == 3:
        R, G, B = img[:,:,0].astype(np.float64), img[:,:,1].astype(np.float64), img[:,:,2].astype(np.float64)
        max_c = np.maximum(np.maximum(R, G), B)
        min_c = np.minimum(np.minimum(R, G), B)
        sat = (max_c - min_c) / (max_c + 1e-12)
        features["saturation_mean"] = float(np.mean(sat))
    else:
        features["saturation_mean"] = 0.0

    return features


INFO_FEATURE_META = {
    "image_entropy": {"description": "Shannon entropy of pixel histogram", "differentiable": False, "batch_computable": True, "family": "info"},
    "color_entropy": {"description": "Mean per-channel entropy", "differentiable": False, "batch_computable": True, "family": "info"},
    "jpeg_compression_ratio": {"description": "JPEG compressed / raw size ratio", "differentiable": False, "batch_computable": True, "family": "info"},
    "png_compression_ratio": {"description": "PNG compressed / raw size ratio", "differentiable": False, "batch_computable": True, "family": "info"},
    "spatial_complexity": {"description": "Mean gradient energy per pixel", "differentiable": True, "batch_computable": True, "family": "info"},
    "fractal_dimension": {"description": "Box-counting fractal dimension", "differentiable": False, "batch_computable": True, "family": "info"},
    "spectral_complexity": {"description": "Fraction of significant frequency components", "differentiable": False, "batch_computable": True, "family": "info"},
    "mutual_info_rgb_channels": {"description": "MI between R and G channels", "differentiable": False, "batch_computable": True, "family": "info"},
    "colorfulness": {"description": "Hasler-Susstrunk colorfulness metric", "differentiable": True, "batch_computable": True, "family": "info"},
    "saturation_mean": {"description": "Mean saturation in HSV-like space", "differentiable": True, "batch_computable": True, "family": "info"},
}
