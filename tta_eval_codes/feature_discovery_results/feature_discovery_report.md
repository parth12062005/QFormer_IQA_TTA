# AGIQA Feature Discovery Report

**Dataset Splits**: test, val
**Total Images Processed**: 4212

## Top 20 Base Features by Spearman Correlation

| Rank | Feature | Spearman | Pearson | MI | TTA Score |
|---|---|---|---|---|---|
| 1 | `spectral_entropy` | 0.6059 | 0.5797 | 0.3103 | 10/10 |
| 2 | `freq_peak_count` | 0.5917 | 0.5312 | 0.3178 | 6/10 |
| 3 | `embed_cls_norm` | 0.5703 | 0.4904 | 0.2644 | 9/10 |
| 4 | `fft_phase_entropy` | 0.5369 | 0.2228 | 0.0264 | 6/10 |
| 5 | `embed_entropy` | -0.5173 | -0.4407 | 0.2260 | 9/10 |
| 6 | `clip_embed_kurtosis` | -0.4823 | -0.6143 | 0.2721 | 6/10 |
| 7 | `clip_embed_sparsity` | 0.4794 | 0.5807 | 0.2718 | 9/10 |
| 8 | `radial_power_intercept` | 0.4658 | 0.4032 | 0.2135 | 6/10 |
| 9 | `spectral_kurtosis` | 0.4198 | 0.1662 | 0.0740 | 6/10 |
| 10 | `spectral_skewness` | 0.4140 | 0.2790 | 0.1623 | 6/10 |
| 11 | `clip_embed_std` | -0.4084 | -0.4781 | 0.2073 | 9/10 |
| 12 | `clip_embed_norm` | -0.4083 | -0.4797 | 0.2101 | 9/10 |
| 13 | `embed_effective_rank` | -0.4064 | -0.4044 | 0.1685 | 6/10 |
| 14 | `ssim_self_similarity` | 0.3976 | 0.3703 | 0.1735 | 5/10 |
| 15 | `fft_magnitude_mean` | 0.3904 | 0.3783 | 0.1533 | 8/10 |
| 16 | `unnatural_sharpness` | 0.3352 | 0.2229 | 0.1176 | 8/10 |
| 17 | `gradient_entropy` | -0.3329 | -0.3234 | 0.1489 | 5/10 |
| 18 | `gradient_magnitude_mean` | -0.3151 | -0.3454 | 0.1372 | 8/10 |
| 19 | `scale_consistency_entropy` | -0.3071 | -0.1873 | 0.0827 | 5/10 |
| 20 | `oversmoothing_score` | 0.3037 | 0.1690 | 0.0836 | 5/10 |

## Top 10 Combo Features

| Rank | Feature | Spearman | Pearson |
|---|---|---|---|
| 1 | `spectral_entropy_mul_embed_cls_norm` | 0.7325 | 0.6687 |
| 2 | `spectral_entropy_sub_embed_entropy` | 0.7112 | 0.6478 |
| 3 | `freq_peak_count_sub_embed_entropy` | 0.6958 | 0.6192 |
| 4 | `spectral_entropy_mul_clip_embed_sparsity` | 0.6745 | 0.6926 |
| 5 | `embed_effective_rank_div_spectral_entropy` | -0.6737 | -0.6644 |
| 6 | `spectral_entropy_div_embed_effective_rank` | 0.6737 | 0.6362 |
| 7 | `spectral_entropy_sub_clip_embed_kurtosis` | 0.6706 | 0.7146 |
| 8 | `spectral_entropy_sub_embed_effective_rank` | 0.6661 | 0.6427 |
| 9 | `embed_cls_norm_mul_radial_power_intercept` | 0.6518 | 0.5733 |
| 10 | `freq_peak_count_mul_embed_cls_norm` | 0.6475 | 0.5675 |

## Top TTA Candidates (Score >= 7)

### `spectral_entropy` (Score: 10/10)
- **Description**: Shannon entropy of normalized magnitude spectrum
- **Spearman vs MOS**: 0.6059
- **Recommendation**: Use as a frequency-domain regularizer (e.g., L2 loss to match source domain spectral flatness).

### `embed_cls_norm` (Score: 9/10)
- **Description**: L2 norm of ViT CLS token
- **Spearman vs MOS**: 0.5703
- **Recommendation**: Use as a representation constraint (e.g., entropy minimization or contrastive alignment).

### `embed_entropy` (Score: 9/10)
- **Description**: Entropy of softmax(CLS token)
- **Spearman vs MOS**: -0.5173
- **Recommendation**: Use as a representation constraint (e.g., entropy minimization or contrastive alignment).

### `clip_embed_sparsity` (Score: 9/10)
- **Description**: L1/(L2*sqrt(N)) of CLIP embedding
- **Spearman vs MOS**: 0.4794
- **Recommendation**: Use as a representation constraint (e.g., entropy minimization or contrastive alignment).

### `clip_embed_std` (Score: 9/10)
- **Description**: Std of CLIP embedding
- **Spearman vs MOS**: -0.4084
- **Recommendation**: Use as a representation constraint (e.g., entropy minimization or contrastive alignment).

### `clip_embed_norm` (Score: 9/10)
- **Description**: L2 norm of CLIP image embedding
- **Spearman vs MOS**: -0.4083
- **Recommendation**: Use as a representation constraint (e.g., entropy minimization or contrastive alignment).

### `fft_magnitude_mean` (Score: 8/10)
- **Description**: Mean of log-magnitude spectrum
- **Spearman vs MOS**: 0.3904
- **Recommendation**: Use as a frequency-domain regularizer (e.g., L2 loss to match source domain spectral flatness).

### `unnatural_sharpness` (Score: 8/10)
- **Description**: Laplacian energy / gradient energy ratio
- **Spearman vs MOS**: 0.3352
- **Recommendation**: Formulate as a direct auxiliary loss term minimized via gradient descent.

### `gradient_magnitude_mean` (Score: 8/10)
- **Description**: Mean Sobel gradient magnitude
- **Spearman vs MOS**: -0.3151
- **Recommendation**: Use as an auxiliary naturalness objective (e.g., minimize distance to expected MSCN distribution).

### `embed_patch_norm_mean` (Score: 8/10)
- **Description**: Mean L2 norm of ViT patch tokens
- **Spearman vs MOS**: 0.2968
- **Recommendation**: Use as a representation constraint (e.g., entropy minimization or contrastive alignment).

### `embed_frobenius_norm` (Score: 8/10)
- **Description**: Frobenius norm of patch embeddings
- **Spearman vs MOS**: 0.2929
- **Recommendation**: Use as a representation constraint (e.g., entropy minimization or contrastive alignment).

### `spatial_complexity` (Score: 8/10)
- **Description**: Mean gradient energy per pixel
- **Spearman vs MOS**: -0.2870
- **Recommendation**: Formulate as a direct auxiliary loss term minimized via gradient descent.

### `embed_patch_similarity_std` (Score: 8/10)
- **Description**: Std of pairwise patch cosine sim
- **Spearman vs MOS**: 0.2810
- **Recommendation**: Use as a representation constraint (e.g., entropy minimization or contrastive alignment).

### `local_contrast_mean` (Score: 8/10)
- **Description**: Mean local_std/local_mean
- **Spearman vs MOS**: -0.2646
- **Recommendation**: Use as an auxiliary naturalness objective (e.g., minimize distance to expected MSCN distribution).

### `gradient_magnitude_std` (Score: 8/10)
- **Description**: Std of gradient magnitude
- **Spearman vs MOS**: -0.2617
- **Recommendation**: Use as an auxiliary naturalness objective (e.g., minimize distance to expected MSCN distribution).

### `spectral_flatness` (Score: 8/10)
- **Description**: Geometric mean / arithmetic mean of spectrum
- **Spearman vs MOS**: 0.2610
- **Recommendation**: Use as a frequency-domain regularizer (e.g., L2 loss to match source domain spectral flatness).

### `fft_magnitude_std` (Score: 8/10)
- **Description**: Std of log-magnitude spectrum
- **Spearman vs MOS**: -0.2591
- **Recommendation**: Use as a frequency-domain regularizer (e.g., L2 loss to match source domain spectral flatness).

### `local_contrast_std` (Score: 8/10)
- **Description**: Std of local contrast
- **Spearman vs MOS**: -0.2302
- **Recommendation**: Use as an auxiliary naturalness objective (e.g., minimize distance to expected MSCN distribution).

### `mid_freq_energy_ratio` (Score: 8/10)
- **Description**: Fraction of energy in 25-75% spectral ring
- **Spearman vs MOS**: -0.2282
- **Recommendation**: Use as a frequency-domain regularizer (e.g., L2 loss to match source domain spectral flatness).

### `low_freq_energy_ratio` (Score: 8/10)
- **Description**: Fraction of energy in inner 25% of spectrum
- **Spearman vs MOS**: 0.2137
- **Recommendation**: Use as a frequency-domain regularizer (e.g., L2 loss to match source domain spectral flatness).

### `mscn_std` (Score: 7/10)
- **Description**: Std of MSCN coefficients
- **Spearman vs MOS**: -0.1933
- **Recommendation**: Use as an auxiliary naturalness objective (e.g., minimize distance to expected MSCN distribution).

### `dino_embed_std` (Score: 7/10)
- **Description**: Std of DINO embedding
- **Spearman vs MOS**: 0.1681
- **Recommendation**: Use as a representation constraint (e.g., entropy minimization or contrastive alignment).

### `dino_embed_norm` (Score: 7/10)
- **Description**: L2 norm of DINO embedding
- **Spearman vs MOS**: 0.1681
- **Recommendation**: Use as a representation constraint (e.g., entropy minimization or contrastive alignment).

### `embed_nuclear_norm` (Score: 7/10)
- **Description**: Nuclear norm (sum of singular values)
- **Spearman vs MOS**: -0.1415
- **Recommendation**: Use as a representation constraint (e.g., entropy minimization or contrastive alignment).

### `mscn_mean` (Score: 7/10)
- **Description**: Mean of MSCN coefficients
- **Spearman vs MOS**: -0.1370
- **Recommendation**: Use as an auxiliary naturalness objective (e.g., minimize distance to expected MSCN distribution).

### `checkerboard_energy` (Score: 7/10)
- **Description**: Energy at Nyquist-adjacent corner frequencies
- **Spearman vs MOS**: 0.1343
- **Recommendation**: Use as a frequency-domain regularizer (e.g., L2 loss to match source domain spectral flatness).

### `colorfulness` (Score: 7/10)
- **Description**: Hasler-Susstrunk colorfulness metric
- **Spearman vs MOS**: -0.1295
- **Recommendation**: Formulate as a direct auxiliary loss term minimized via gradient descent.

### `freq_sparsity` (Score: 7/10)
- **Description**: L1/(L2*sqrt(N)) sparsity of magnitude spectrum
- **Spearman vs MOS**: -0.1059
- **Recommendation**: Use as a frequency-domain regularizer (e.g., L2 loss to match source domain spectral flatness).

### `saturation_mean` (Score: 7/10)
- **Description**: Mean saturation in HSV-like space
- **Spearman vs MOS**: -0.1031
- **Recommendation**: Formulate as a direct auxiliary loss term minimized via gradient descent.

### `laplacian_var` (Score: 7/10)
- **Description**: Variance of Laplacian (sharpness)
- **Spearman vs MOS**: -0.1022
- **Recommendation**: Use as an auxiliary naturalness objective (e.g., minimize distance to expected MSCN distribution).

### `laplacian_energy` (Score: 7/10)
- **Description**: Mean squared Laplacian response
- **Spearman vs MOS**: -0.1022
- **Recommendation**: Use as an auxiliary naturalness objective (e.g., minimize distance to expected MSCN distribution).

### `clip_embed_mean` (Score: 7/10)
- **Description**: Mean of CLIP embedding
- **Spearman vs MOS**: 0.0935
- **Recommendation**: Use as a representation constraint (e.g., entropy minimization or contrastive alignment).

### `clip_embed_entropy` (Score: 7/10)
- **Description**: Entropy of softmax(CLIP embedding)
- **Spearman vs MOS**: -0.0883
- **Recommendation**: Use as a representation constraint (e.g., entropy minimization or contrastive alignment).

### `dct_energy_ratio` (Score: 7/10)
- **Description**: Energy in low-frequency DCT quadrant
- **Spearman vs MOS**: 0.0813
- **Recommendation**: Use as a frequency-domain regularizer (e.g., L2 loss to match source domain spectral flatness).

### `hf_lf_ratio` (Score: 7/10)
- **Description**: High frequency / low frequency energy ratio
- **Spearman vs MOS**: -0.0795
- **Recommendation**: Use as a frequency-domain regularizer (e.g., L2 loss to match source domain spectral flatness).

### `high_freq_energy_ratio` (Score: 7/10)
- **Description**: Fraction of energy in outer 25% of spectrum
- **Spearman vs MOS**: -0.0773
- **Recommendation**: Use as a frequency-domain regularizer (e.g., L2 loss to match source domain spectral flatness).

### `dino_embed_mean` (Score: 7/10)
- **Description**: Mean of DINO embedding
- **Spearman vs MOS**: -0.0457
- **Recommendation**: Use as a representation constraint (e.g., entropy minimization or contrastive alignment).

### `dino_embed_sparsity` (Score: 7/10)
- **Description**: L1/(L2*sqrt(N)) of DINO embedding
- **Spearman vs MOS**: 0.0324
- **Recommendation**: Use as a representation constraint (e.g., entropy minimization or contrastive alignment).

### `fft_phase_std` (Score: 7/10)
- **Description**: Std of FFT phase angles
- **Spearman vs MOS**: -0.0159
- **Recommendation**: Use as a frequency-domain regularizer (e.g., L2 loss to match source domain spectral flatness).

### `dino_embed_entropy` (Score: 7/10)
- **Description**: Entropy of softmax(DINO embedding)
- **Spearman vs MOS**: -0.0138
- **Recommendation**: Use as a representation constraint (e.g., entropy minimization or contrastive alignment).

### `embed_patch_similarity_mean` (Score: 7/10)
- **Description**: Mean pairwise cosine sim of patches
- **Spearman vs MOS**: 0.0137
- **Recommendation**: Use as a representation constraint (e.g., entropy minimization or contrastive alignment).

### `dc_component_ratio` (Score: 7/10)
- **Description**: DC component / total spectral energy
- **Spearman vs MOS**: 0.0045
- **Recommendation**: Use as a frequency-domain regularizer (e.g., L2 loss to match source domain spectral flatness).

### `embed_patch_norm_std` (Score: 7/10)
- **Description**: Std of ViT patch L2 norms
- **Spearman vs MOS**: -0.0031
- **Recommendation**: Use as a representation constraint (e.g., entropy minimization or contrastive alignment).

