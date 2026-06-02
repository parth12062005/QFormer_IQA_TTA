"""
Multi-scale features for image quality assessment.
Computes features across 1x, 1/2x, 1/4x, and 1/8x scales and measures consistency/slopes.
"""
import numpy as np
from PIL import Image
from scipy import ndimage


def _to_gray(img):
    if img.ndim == 3:
        return np.dot(img[..., :3].astype(np.float64), [0.2989, 0.5870, 0.1140])
    return img.astype(np.float64)


def _entropy_hist(data, bins=256):
    hist, _ = np.histogram(data.ravel(), bins=bins, density=True)
    hist = hist / (hist.sum() + 1e-12)
    return float(-np.sum(hist[hist > 0] * np.log2(hist[hist > 0])))


def compute_multiscale_features(img):
    """
    Compute multiscale features.
    
    Args:
        img: numpy array HWC uint8
    
    Returns:
        dict[str, float]
    """
    features = {}
    scales = [1, 2, 4, 8]
    img_pil = Image.fromarray(img)
    
    entropies = []
    gradients = []
    hf_ratios = []
    contrasts = []
    laplacians = []
    mscn_kurtoses = []
    colorfulnesses = []
    spectral_flatnesses = []
    
    for scale in scales:
        if scale == 1:
            scaled_img = img
        else:
            w, h = img_pil.size
            if w // scale < 8 or h // scale < 8:
                break
            scaled_pil = img_pil.resize((w // scale, h // scale), Image.BICUBIC)
            scaled_img = np.array(scaled_pil)
            
        gray = _to_gray(scaled_img)
        
        # Entropy
        entropies.append(_entropy_hist(gray))
        
        # Gradient energy
        gx = ndimage.sobel(gray, axis=1)
        gy = ndimage.sobel(gray, axis=0)
        gradients.append(np.mean(gx**2 + gy**2))
        
        # High frequency energy ratio
        fft = np.fft.fft2(gray)
        power = np.abs(np.fft.fftshift(fft))**2
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        r = np.sqrt((X - cx)**2 + (Y - cy)**2)
        max_r = min(cx, cy)
        hf_mask = r > max_r * 0.75
        hf_ratios.append(np.sum(power[hf_mask]) / (np.sum(power) + 1e-12))
        
        # Local contrast
        mu = ndimage.uniform_filter(gray, size=7)
        sq = ndimage.uniform_filter(gray ** 2, size=7)
        sigma = np.sqrt(np.maximum(sq - mu**2, 0))
        contrasts.append(np.mean(sigma / (mu + 1e-12)))
        
        # Laplacian variance
        lap = ndimage.laplace(gray)
        laplacians.append(np.var(lap))
        
        # MSCN kurtosis
        mscn = (gray - mu) / (sigma + 1.0)
        mscn_flat = mscn.ravel()
        from scipy import stats
        mscn_kurtoses.append(stats.kurtosis(mscn_flat, fisher=True))
        
        # Colorfulness
        if scaled_img.ndim == 3:
            R, G, B = scaled_img[...,0].astype(np.float64), scaled_img[...,1].astype(np.float64), scaled_img[...,2].astype(np.float64)
            rg = R - G
            yb = 0.5 * (R + G) - B
            c_val = np.sqrt(np.std(rg)**2 + np.std(yb)**2) + 0.3 * np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)
            colorfulnesses.append(c_val)
        else:
            colorfulnesses.append(0.0)
            
        # Spectral flatness
        mag_flat = np.sqrt(power).ravel()
        log_mag = np.log(mag_flat + 1e-12)
        geo_mean = np.exp(np.mean(log_mag))
        arith_mean = np.mean(mag_flat) + 1e-12
        spectral_flatnesses.append(geo_mean / arith_mean)

    # Compute slopes and consistencies if we have enough scales
    if len(entropies) >= 3:
        x = np.log(scales[:len(entropies)])
        
        def _slope(y):
            if np.all(np.isfinite(y)):
                return float(np.polyfit(x, y, 1)[0])
            return 0.0
            
        features["multiscale_entropy_slope"] = _slope(entropies)
        features["multiscale_gradient_slope"] = _slope(np.log(np.array(gradients) + 1e-12))
        features["multiscale_fft_slope"] = _slope(hf_ratios)
        features["multiscale_contrast_slope"] = _slope(contrasts)
        features["multiscale_laplacian_slope"] = _slope(np.log(np.array(laplacians) + 1e-12))
        features["multiscale_mscn_slope"] = _slope(mscn_kurtoses)
        features["multiscale_colorfulness_slope"] = _slope(colorfulnesses)
        features["multiscale_spectral_flatness_slope"] = _slope(spectral_flatnesses)
        
        features["scale_consistency_entropy"] = float(np.std(entropies))
        features["scale_consistency_gradient"] = float(np.std(np.log(np.array(gradients) + 1e-12)))
        features["scale_consistency_fft"] = float(np.std(spectral_flatnesses))
    else:
        for k in ["multiscale_entropy_slope", "multiscale_gradient_slope", "multiscale_fft_slope",
                  "multiscale_contrast_slope", "multiscale_laplacian_slope", "multiscale_mscn_slope",
                  "multiscale_colorfulness_slope", "multiscale_spectral_flatness_slope",
                  "scale_consistency_entropy", "scale_consistency_gradient", "scale_consistency_fft"]:
            features[k] = 0.0

    return features


MULTISCALE_FEATURE_META = {
    "multiscale_entropy_slope": {"description": "Slope of entropy across scales", "differentiable": False, "batch_computable": True, "family": "multiscale"},
    "multiscale_gradient_slope": {"description": "Slope of gradient energy across scales", "differentiable": False, "batch_computable": True, "family": "multiscale"},
    "multiscale_fft_slope": {"description": "Slope of high-freq energy ratio across scales", "differentiable": False, "batch_computable": True, "family": "multiscale"},
    "multiscale_contrast_slope": {"description": "Slope of local contrast across scales", "differentiable": False, "batch_computable": True, "family": "multiscale"},
    "scale_consistency_entropy": {"description": "Std of entropy across scales", "differentiable": False, "batch_computable": True, "family": "multiscale"},
    "scale_consistency_gradient": {"description": "Std of gradient energy across scales", "differentiable": False, "batch_computable": True, "family": "multiscale"},
    "scale_consistency_fft": {"description": "Std of spectral flatness across scales", "differentiable": False, "batch_computable": True, "family": "multiscale"},
    "multiscale_laplacian_slope": {"description": "Slope of Laplacian variance across scales", "differentiable": False, "batch_computable": True, "family": "multiscale"},
    "multiscale_mscn_slope": {"description": "Slope of MSCN kurtosis across scales", "differentiable": False, "batch_computable": True, "family": "multiscale"},
    "multiscale_colorfulness_slope": {"description": "Slope of colorfulness across scales", "differentiable": False, "batch_computable": True, "family": "multiscale"},
    "multiscale_spectral_flatness_slope": {"description": "Slope of spectral flatness across scales", "differentiable": False, "batch_computable": True, "family": "multiscale"},
}
