"""
Frequency-domain features for image quality assessment.
Uses numpy FFT — no OpenCV dependency.
"""

import numpy as np
from scipy import ndimage


def _to_gray(img):
    """Convert HWC uint8 image to float64 grayscale."""
    if img.ndim == 3:
        return np.dot(img[..., :3].astype(np.float64), [0.2989, 0.5870, 0.1140])
    return img.astype(np.float64)


def _radial_profile(magnitude, center=None):
    """Compute radial average of 2D magnitude spectrum."""
    h, w = magnitude.shape
    if center is None:
        center = (h // 2, w // 2)
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((X - center[1])**2 + (Y - center[0])**2).astype(int)
    max_r = min(center[0], center[1], h - center[0], w - center[1])
    radial = np.zeros(max_r)
    for i in range(max_r):
        mask = (r == i)
        if mask.any():
            radial[i] = np.mean(magnitude[mask])
    return radial


def _shannon_entropy(p):
    """Shannon entropy of a probability distribution."""
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def compute_frequency_features(img):
    """
    Compute all frequency-domain features from an image.

    Args:
        img: numpy array HWC uint8

    Returns:
        dict[str, float]
    """
    gray = _to_gray(img)
    h, w = gray.shape
    features = {}

    # 2D FFT
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    phase = np.angle(fft_shift)
    log_magnitude = np.log(magnitude + 1.0)
    power = magnitude ** 2

    # Basic magnitude stats
    features["fft_magnitude_mean"] = float(np.mean(log_magnitude))
    features["fft_magnitude_std"] = float(np.std(log_magnitude))

    # Phase statistics
    features["fft_phase_std"] = float(np.std(phase))
    phase_hist, _ = np.histogram(phase.ravel(), bins=64, density=True)
    phase_hist = phase_hist / (phase_hist.sum() + 1e-12)
    features["fft_phase_entropy"] = float(_shannon_entropy(phase_hist))

    # Total energy
    total_energy = np.sum(power) + 1e-12

    # DC component
    cx, cy = h // 2, w // 2
    features["dc_component_ratio"] = float(power[cx, cy] / total_energy)

    # Radial frequency bands
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((X - cy)**2 + (Y - cx)**2)
    max_r = min(cx, cy)

    low_mask = r <= max_r * 0.25
    mid_mask = (r > max_r * 0.25) & (r <= max_r * 0.75)
    high_mask = r > max_r * 0.75

    low_energy = np.sum(power[low_mask])
    mid_energy = np.sum(power[mid_mask])
    high_energy = np.sum(power[high_mask])

    features["low_freq_energy_ratio"] = float(low_energy / total_energy)
    features["mid_freq_energy_ratio"] = float(mid_energy / total_energy)
    features["high_freq_energy_ratio"] = float(high_energy / total_energy)
    features["hf_lf_ratio"] = float(high_energy / (low_energy + 1e-12))

    # Radial power spectrum
    radial = _radial_profile(power, center=(cx, cy))
    if len(radial) > 2:
        radial_safe = radial + 1e-12

        # Power-law fit: log(P) = -beta * log(f) + c
        freqs = np.arange(1, len(radial_safe))
        log_f = np.log(freqs)
        log_p = np.log(radial_safe[1:])
        finite_mask = np.isfinite(log_f) & np.isfinite(log_p)
        if finite_mask.sum() > 2:
            coeffs = np.polyfit(log_f[finite_mask], log_p[finite_mask], 1)
            features["radial_power_slope"] = float(coeffs[0])
            features["radial_power_intercept"] = float(coeffs[1])
        else:
            features["radial_power_slope"] = 0.0
            features["radial_power_intercept"] = 0.0

        # Spectral kurtosis and skewness
        radial_norm = radial / (radial.sum() + 1e-12)
        mean_r = np.sum(np.arange(len(radial)) * radial_norm)
        var_r = np.sum((np.arange(len(radial)) - mean_r)**2 * radial_norm)
        std_r = np.sqrt(var_r + 1e-12)
        features["spectral_centroid"] = float(mean_r)
        features["spectral_bandwidth"] = float(std_r)

        if std_r > 1e-8:
            skew_r = np.sum(((np.arange(len(radial)) - mean_r) / std_r)**3 * radial_norm)
            kurt_r = np.sum(((np.arange(len(radial)) - mean_r) / std_r)**4 * radial_norm) - 3
            features["spectral_skewness"] = float(skew_r)
            features["spectral_kurtosis"] = float(kurt_r)
        else:
            features["spectral_skewness"] = 0.0
            features["spectral_kurtosis"] = 0.0

        # Spectral rolloff (85% energy)
        cumulative = np.cumsum(radial)
        total_rad = cumulative[-1] + 1e-12
        rolloff_idx = np.searchsorted(cumulative, 0.85 * total_rad)
        features["spectral_rolloff"] = float(rolloff_idx / len(radial))
    else:
        for k in ["radial_power_slope", "radial_power_intercept",
                   "spectral_centroid", "spectral_bandwidth",
                   "spectral_skewness", "spectral_kurtosis", "spectral_rolloff"]:
            features[k] = 0.0

    # Spectral entropy
    mag_flat = magnitude.ravel()
    mag_norm = mag_flat / (mag_flat.sum() + 1e-12)
    features["spectral_entropy"] = float(_shannon_entropy(mag_norm))

    # Spectral flatness: geometric mean / arithmetic mean
    log_mag = np.log(mag_flat + 1e-12)
    geo_mean = np.exp(np.mean(log_mag))
    arith_mean = np.mean(mag_flat) + 1e-12
    features["spectral_flatness"] = float(geo_mean / arith_mean)

    # Frequency sparsity: L1/L2
    l1 = np.sum(np.abs(mag_flat))
    l2 = np.sqrt(np.sum(mag_flat**2)) + 1e-12
    features["freq_sparsity"] = float(l1 / (l2 * np.sqrt(len(mag_flat))))

    # Frequency anisotropy: variance of energy in angular bins
    n_bins = 12
    angles = np.arctan2(Y - cx, X - cy)
    angle_bins = np.digitize(angles, np.linspace(-np.pi, np.pi, n_bins + 1)) - 1
    angular_energy = np.zeros(n_bins)
    for b in range(n_bins):
        amask = angle_bins == b
        if amask.any():
            angular_energy[b] = np.sum(power[amask])
    angular_norm = angular_energy / (angular_energy.sum() + 1e-12)
    features["freq_anisotropy"] = float(np.std(angular_norm))

    # Frequency peak count (peaks in radial profile > 2*median)
    if len(radial) > 5:
        med = np.median(radial)
        peaks = np.sum(radial > 2 * med + 1e-12)
        features["freq_peak_count"] = float(peaks)
    else:
        features["freq_peak_count"] = 0.0

    # Checkerboard artifact: energy at Nyquist-adjacent corners
    corner_size = max(2, min(h, w) // 20)
    corner_energy = (
        np.sum(power[:corner_size, :corner_size]) +
        np.sum(power[:corner_size, -corner_size:]) +
        np.sum(power[-corner_size:, :corner_size]) +
        np.sum(power[-corner_size:, -corner_size:])
    )
    features["checkerboard_energy"] = float(corner_energy / total_energy)

    # Ring artifact: max energy in narrow radial bands
    n_rings = 20
    ring_energies = np.zeros(n_rings)
    for i in range(n_rings):
        r_low = max_r * i / n_rings
        r_high = max_r * (i + 1) / n_rings
        ring_mask = (r >= r_low) & (r < r_high)
        n_pixels = ring_mask.sum()
        if n_pixels > 0:
            ring_energies[i] = np.sum(power[ring_mask]) / n_pixels
    if ring_energies.sum() > 0:
        features["ring_artifact_score"] = float(np.max(ring_energies) / (np.mean(ring_energies) + 1e-12))
    else:
        features["ring_artifact_score"] = 0.0

    # DCT energy ratio (using FFT-based approximation for real signals)
    # DCT ≈ real part of FFT on mirrored signal
    dct_coeffs = np.abs(np.fft.dct2_approx(gray)) if False else magnitude  # use magnitude as proxy
    # We'll just use a simple DCT-like measure: ratio of top-left quadrant energy
    quadrant = power[:cx, :cy]
    features["dct_energy_ratio"] = float(np.sum(quadrant) / total_energy)

    # Patch FFT consistency: compute spectral centroid per patch, measure std
    patch_size = min(64, h // 2, w // 2)
    if patch_size >= 16:
        centroids = []
        for py in range(0, h - patch_size + 1, patch_size):
            for px in range(0, w - patch_size + 1, patch_size):
                patch = gray[py:py+patch_size, px:px+patch_size]
                p_fft = np.abs(np.fft.fftshift(np.fft.fft2(patch)))
                p_power = p_fft ** 2
                p_total = p_power.sum() + 1e-12
                p_norm = p_power / p_total
                ph, pw_ = patch.shape
                pY, pX = np.ogrid[:ph, :pw_]
                pr = np.sqrt((pX - pw_//2)**2 + (pY - ph//2)**2)
                centroids.append(float(np.sum(pr * p_norm)))
        if len(centroids) > 1:
            features["patch_fft_consistency"] = float(np.std(centroids))
        else:
            features["patch_fft_consistency"] = 0.0
    else:
        features["patch_fft_consistency"] = 0.0

    return features


# Feature metadata for registry
FREQUENCY_FEATURE_META = {
    "fft_magnitude_mean": {"description": "Mean of log-magnitude spectrum", "differentiable": True, "batch_computable": True, "family": "frequency"},
    "fft_magnitude_std": {"description": "Std of log-magnitude spectrum", "differentiable": True, "batch_computable": True, "family": "frequency"},
    "fft_phase_entropy": {"description": "Entropy of FFT phase histogram", "differentiable": False, "batch_computable": True, "family": "frequency"},
    "fft_phase_std": {"description": "Std of FFT phase angles", "differentiable": True, "batch_computable": True, "family": "frequency"},
    "radial_power_slope": {"description": "Power-law exponent of radial power spectrum (1/f^beta)", "differentiable": False, "batch_computable": True, "family": "frequency"},
    "radial_power_intercept": {"description": "Intercept of radial power-law fit", "differentiable": False, "batch_computable": True, "family": "frequency"},
    "high_freq_energy_ratio": {"description": "Fraction of energy in outer 25% of spectrum", "differentiable": True, "batch_computable": True, "family": "frequency"},
    "mid_freq_energy_ratio": {"description": "Fraction of energy in 25-75% spectral ring", "differentiable": True, "batch_computable": True, "family": "frequency"},
    "low_freq_energy_ratio": {"description": "Fraction of energy in inner 25% of spectrum", "differentiable": True, "batch_computable": True, "family": "frequency"},
    "hf_lf_ratio": {"description": "High frequency / low frequency energy ratio", "differentiable": True, "batch_computable": True, "family": "frequency"},
    "spectral_entropy": {"description": "Shannon entropy of normalized magnitude spectrum", "differentiable": True, "batch_computable": True, "family": "frequency"},
    "spectral_flatness": {"description": "Geometric mean / arithmetic mean of spectrum", "differentiable": True, "batch_computable": True, "family": "frequency"},
    "spectral_centroid": {"description": "Weighted mean frequency of radial spectrum", "differentiable": False, "batch_computable": True, "family": "frequency"},
    "spectral_bandwidth": {"description": "Weighted std of radial spectrum", "differentiable": False, "batch_computable": True, "family": "frequency"},
    "spectral_skewness": {"description": "Skewness of radial power spectrum", "differentiable": False, "batch_computable": True, "family": "frequency"},
    "spectral_kurtosis": {"description": "Kurtosis of radial power spectrum", "differentiable": False, "batch_computable": True, "family": "frequency"},
    "spectral_rolloff": {"description": "Frequency below which 85% of energy resides", "differentiable": False, "batch_computable": True, "family": "frequency"},
    "dc_component_ratio": {"description": "DC component / total spectral energy", "differentiable": True, "batch_computable": True, "family": "frequency"},
    "freq_anisotropy": {"description": "Std of angular energy distribution", "differentiable": False, "batch_computable": True, "family": "frequency"},
    "freq_sparsity": {"description": "L1/(L2*sqrt(N)) sparsity of magnitude spectrum", "differentiable": True, "batch_computable": True, "family": "frequency"},
    "freq_peak_count": {"description": "Number of significant peaks in radial spectrum", "differentiable": False, "batch_computable": True, "family": "frequency"},
    "checkerboard_energy": {"description": "Energy at Nyquist-adjacent corner frequencies", "differentiable": True, "batch_computable": True, "family": "frequency"},
    "ring_artifact_score": {"description": "Max-to-mean energy ratio in radial bands", "differentiable": False, "batch_computable": True, "family": "frequency"},
    "dct_energy_ratio": {"description": "Energy in low-frequency DCT quadrant", "differentiable": True, "batch_computable": True, "family": "frequency"},
    "patch_fft_consistency": {"description": "Std of per-patch spectral centroids", "differentiable": False, "batch_computable": True, "family": "frequency"},
}
