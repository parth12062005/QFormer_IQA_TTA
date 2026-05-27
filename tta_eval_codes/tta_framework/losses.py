"""
TTA Loss Functions for IQA.

Each loss is a callable class inheriting from TTALoss.
All losses receive a context dict and return a scalar tensor.

Context dict keys (populated by the engine as needed):
    proj_feats        : (B, proj_dim) — projected Q-Former features (original images)
    proj_feats_weak   : (B, proj_dim) — from weak augmentation  (only if requires_augmentations)
    proj_feats_strong : (B, proj_dim) — from strong augmentation (only if requires_augmentations)
    predictions       : (B,) — regressor predictions (detached, for GC clustering)
    vgg_feats         : (B, 512) — VGG-16 features (only if requires_vgg)
    device            : torch.device
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
#  Base class
# ---------------------------------------------------------------------------
class TTALoss:
    """Base class for TTA auxiliary losses."""
    name: str = "base"
    requires_augmentations: bool = False   # needs weak/strong augmented features
    requires_vgg: bool = False             # needs VGG-16 features

    def __call__(self, ctx: dict) -> torch.Tensor:
        raise NotImplementedError


# ---------------------------------------------------------------------------
#  1) Group Contrastive (GC) Loss  — Roy et al. ICCV 2023
# ---------------------------------------------------------------------------
class GCLoss(TTALoss):
    """
    Groups batch into high/low quality clusters using the model's own
    predicted scores (top/bottom p%), then applies InfoNCE contrastive loss.
    """
    name = "gc"
    requires_augmentations = False
    requires_vgg = False

    def __init__(self, p: float = 0.25, temperature: float = 0.1):
        self.p = p
        self.temperature = temperature

    def __call__(self, ctx: dict) -> torch.Tensor:
        feats = ctx["proj_feats"]              # (B, D), already L2-normed
        preds = ctx["predictions"]             # (B,)
        device = ctx["device"]
        B = feats.size(0)
        k = max(2, int(B * self.p))

        if B < 4:
            return torch.tensor(0.0, device=device, requires_grad=True)

        idx = torch.argsort(preds)
        low_feats  = F.normalize(feats[idx[:k]], dim=-1)
        high_feats = F.normalize(feats[idx[-k:]], dim=-1)

        loss = 0.0
        for i in range(k):
            # anchor = high_feats[i]
            pos_sim = torch.exp(torch.matmul(high_feats[i], high_feats.t()) / self.temperature)
            neg_sim = torch.exp(torch.matmul(high_feats[i], low_feats.t()) / self.temperature)
            pos_sum = pos_sim.sum() - pos_sim[i]  # exclude self
            loss += -torch.log(pos_sum / (pos_sum + neg_sim.sum() + 1e-8))

            # anchor = low_feats[i]
            pos_sim_l = torch.exp(torch.matmul(low_feats[i], low_feats.t()) / self.temperature)
            neg_sim_l = torch.exp(torch.matmul(low_feats[i], high_feats.t()) / self.temperature)
            pos_sum_l = pos_sim_l.sum() - pos_sim_l[i]
            loss += -torch.log(pos_sum_l / (pos_sum_l + neg_sim_l.sum() + 1e-8))

        return loss / (2 * k)


# ---------------------------------------------------------------------------
#  2) Rank Loss  — Roy et al. ICCV 2023
# ---------------------------------------------------------------------------
class RankLoss(TTALoss):
    """
    Enforces that heavily distorted images are farther from originals than
    lightly distorted ones:  dist(strong, orig) > dist(weak, orig).
    Uses BCE formulation: P(d_strong >= d_weak) should be 1.
    """
    name = "rank"
    requires_augmentations = True
    requires_vgg = False

    def __call__(self, ctx: dict) -> torch.Tensor:
        feat_orig   = ctx["proj_feats"]          # (B, D)
        feat_weak   = ctx["proj_feats_weak"]     # (B, D)
        feat_strong = ctx["proj_feats_strong"]   # (B, D)

        dist_weak   = F.pairwise_distance(feat_weak, feat_orig, p=2)
        dist_strong = F.pairwise_distance(feat_strong, feat_orig, p=2)

        target = torch.ones_like(dist_strong)
        return F.binary_cross_entropy_with_logits(dist_strong - dist_weak, target)


# ---------------------------------------------------------------------------
#  3) Feature Affinity-based Group Contrastive (FAGC) Loss  — Kapoor et al. ICME 2025
# ---------------------------------------------------------------------------
class FAGCLoss(TTALoss):
    """
    Uses VGG-16 feature affinity (cosine similarity with the highest-quality
    image) to cluster the batch, then applies contrastive loss.
    More robust than GC under distribution shift because clustering is
    based on perceptual feature similarity rather than predicted scores.
    """
    name = "fagc"
    requires_augmentations = False
    requires_vgg = True

    def __init__(self, temperature: float = 0.5):
        self.temperature = temperature

    def _cluster_by_vgg_affinity(self, vgg_feats, predictions):
        """
        Cluster using VGG-16 features:
        1. Find the image with highest predicted quality (imax).
        2. Compute cosine similarity of all images with imax.
        3. Top half → high-quality cluster (C1), bottom half → low-quality (C2).
        """
        B = vgg_feats.size(0)
        if B < 2:
            return torch.arange(B), torch.arange(B)

        # Find highest-quality image according to base model
        imax = torch.argmax(predictions)
        f_imax = vgg_feats[imax].unsqueeze(0)  # (1, D)

        # Cosine affinity with all other images
        affinities = F.cosine_similarity(f_imax, vgg_feats, dim=-1)  # (B,)

        # Split into two clusters by median affinity
        half = B // 2
        sorted_idx = torch.argsort(affinities, descending=True)
        c1_idx = sorted_idx[:half]   # high affinity → high quality
        c2_idx = sorted_idx[half:]   # low affinity → low quality

        if len(c1_idx) == 0 or len(c2_idx) == 0:
            c1_idx = torch.arange(0, half)
            c2_idx = torch.arange(half, B)

        return c1_idx, c2_idx

    def __call__(self, ctx: dict) -> torch.Tensor:
        proj_feats  = ctx["proj_feats"]     # (B, D)
        vgg_feats   = ctx["vgg_feats"]      # (B, 512)
        predictions = ctx["predictions"]    # (B,)
        device      = ctx["device"]

        c1_idx, c2_idx = self._cluster_by_vgg_affinity(vgg_feats, predictions)

        loss = 0.0
        valid_clusters = 0

        for cluster_idx in [c1_idx, c2_idx]:
            n = len(cluster_idx)
            if n > 1:
                feats = proj_feats[cluster_idx]
                sim_matrix = torch.mm(feats, feats.t()) / self.temperature
                sim_matrix.fill_diagonal_(-1e9)

                other_idx = c2_idx if cluster_idx is c1_idx else c1_idx
                if len(other_idx) > 0:
                    cross_sim = torch.mm(feats, proj_feats[other_idx].t()) / self.temperature
                    logits = torch.cat([sim_matrix, cross_sim], dim=1)
                else:
                    logits = sim_matrix

                labels = torch.arange(n, device=device)
                loss += F.cross_entropy(logits, labels)
                valid_clusters += 1

        if valid_clusters > 0:
            return loss / valid_clusters
        return torch.tensor(0.0, device=device, requires_grad=True)


# ---------------------------------------------------------------------------
#  4) Adaptive Rank Loss  — Kapoor et al. ICME 2025
# ---------------------------------------------------------------------------
class AdaptiveRankLoss(TTALoss):
    """
    Like Rank Loss but with an adaptive γ term.
    When the base model already distinguishes distortion levels well
    (counter1 >= counter2), rank loss impact is reduced to a constant λ.
    Otherwise, γ = d_high - d_low (actual gap).

    From the paper (Eq 4-6):
      γ = λ           if counter1 >= counter2
      γ = d_high - d_low   otherwise

      y = exp(γ) / (1 + exp(γ))
      L = -Σ log(y)
    """
    name = "adaptive_rank"
    requires_augmentations = True
    requires_vgg = False

    def __init__(self, lam: float = 1.0):
        self.lam = lam
        self.counter1 = 0  # times rank is satisfied (d_high > d_low)
        self.counter2 = 0  # times rank is violated  (d_high <= d_low)

    def reset_counters(self):
        self.counter1 = 0
        self.counter2 = 0

    def __call__(self, ctx: dict) -> torch.Tensor:
        feat_orig   = ctx["proj_feats"]
        feat_weak   = ctx["proj_feats_weak"]
        feat_strong = ctx["proj_feats_strong"]

        d_low  = F.pairwise_distance(feat_weak, feat_orig, p=2)    # mild distortion
        d_high = F.pairwise_distance(feat_strong, feat_orig, p=2)  # heavy distortion

        # Update counters
        satisfied = (d_high > d_low).sum().item()
        violated  = (d_high <= d_low).sum().item()
        self.counter1 += satisfied
        self.counter2 += violated

        # Compute adaptive gamma
        if self.counter1 >= self.counter2:
            gamma = torch.full_like(d_high, self.lam)
        else:
            gamma = d_high - d_low

        # y = sigmoid(gamma) = P(d_high >= d_low)
        y = torch.sigmoid(gamma)
        # L = -mean(log(y))
        loss = -torch.log(y + 1e-8).mean()
        return loss


# ---------------------------------------------------------------------------
#  Registry for CLI lookup
# ---------------------------------------------------------------------------
LOSS_REGISTRY = {
    "gc":            GCLoss,
    "rank":          RankLoss,
    "fagc":          FAGCLoss,
    "adaptive_rank": AdaptiveRankLoss,
}
