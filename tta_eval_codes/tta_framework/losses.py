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
#      Ported from original TTA-IQA GroupContrastiveLoss (util.py)
# ---------------------------------------------------------------------------
class GCLoss(TTALoss):
    """
    SimCLR-style group contrastive loss matching the original TTA-IQA.
    Groups batch into high/low quality clusters using predicted scores,
    concatenates them, and maximizes within-cluster similarity while
    minimizing cross-cluster similarity.
    """
    name = "gc"
    requires_augmentations = False
    requires_vgg = False

    def __init__(self, p: float = 0.25, temperature: float = 0.5):
        self.p = p
        self.temperature = temperature

    def __call__(self, ctx: dict) -> torch.Tensor:
        feats = ctx["proj_feats"]              # (B, D)
        preds = ctx["predictions"]             # (B,)
        device = ctx["device"]
        B = feats.size(0)
        k = max(2, int(B * self.p))

        if B < 4:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Cluster: bottom k = low quality, top k = high quality
        idx = torch.argsort(preds)
        emb_i = feats[idx[:k]]     # low quality
        emb_j = feats[idx[-k:]]    # high quality
        n_i, n_j = emb_i.size(0), emb_j.size(0)

        # Normalize
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        # Concatenate and compute full similarity matrix
        representations = torch.cat([z_i, z_j], dim=0)  # (2k, D)
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=2
        )  # (2k, 2k)

        # Within-cluster positive similarity (average, excluding self)
        # pos block for z_i is top-left (n_i × n_j), for z_j is bottom-right
        pos_sim_ij = similarity_matrix[:n_i, :n_j]
        positives_mask_ij = (~torch.eye(n_i, n_j, dtype=bool, device=device)).float()
        sim_ij = torch.sum(pos_sim_ij * positives_mask_ij, dim=1) / max(n_j - 1, 1)

        pos_sim_ji = similarity_matrix[n_i:, n_j:]
        positives_mask_ji = (~torch.eye(n_j, n_i, dtype=bool, device=device)).float()
        sim_ji = torch.sum(pos_sim_ji * positives_mask_ji, dim=1) / max(n_i - 1, 1)

        positives = torch.cat([sim_ij, sim_ji], dim=0)  # (2k,)

        # Negatives mask: block off within-cluster pairs
        total = n_i + n_j
        negatives_mask = torch.ones(total, total, dtype=bool, device=device)
        negatives_mask[:n_i, :n_j] = False   # block top-left
        negatives_mask[n_i:, n_j:] = False   # block bottom-right
        negatives_mask = negatives_mask.float()

        # Loss
        nominator = torch.exp(positives / self.temperature)
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = torch.sum(
            nominator / (nominator + torch.sum(denominator, dim=1) + 1e-8)
        ) / total
        loss = -torch.log(loss_partial + 1e-8)

        return loss


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
#      Matches Eq. 2 of the paper:
#        L = -log( exp(Sim(li,lj)/τ) / Σ_k∈C_opp exp(Sim(li,lk)/τ) )
#      Sim = scaled dot-product attention: (a·b)/√d
# ---------------------------------------------------------------------------
class FAGCLoss(TTALoss):
    """
    Uses VGG-16 feature affinity (cosine similarity with the highest-quality
    image) to cluster the batch into C1 (high) and C2 (low), then applies
    a per-anchor-per-positive contrastive loss.

    For each anchor i in C1 and each positive j in C1 (j≠i):
        L = -log( exp(Sim(li,lj)/τ) / Σ_{k∈C2} exp(Sim(li,lk)/τ) )
    Same is computed for C2 anchors with C1 as negatives.
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

        # Cosine affinity with all other images (Eq. 1)
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

    def _scaled_dot_product_sim(self, a, b):
        """Scaled dot-product attention similarity: (a·b)/√d"""
        d = a.size(-1)
        return torch.mm(a, b.t()) / (d ** 0.5)

    def __call__(self, ctx: dict) -> torch.Tensor:
        proj_feats  = ctx["proj_feats"]     # (B, D)
        vgg_feats   = ctx["vgg_feats"]      # (B, 512)
        predictions = ctx["predictions"]    # (B,)
        device      = ctx["device"]

        c1_idx, c2_idx = self._cluster_by_vgg_affinity(vgg_feats, predictions)

        c1_feats = proj_feats[c1_idx]  # (|C1|, D)
        c2_feats = proj_feats[c2_idx]  # (|C2|, D)
        n1, n2 = c1_feats.size(0), c2_feats.size(0)

        if n1 < 2 or n2 < 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = 0.0
        count = 0

        # --- Anchors in C1, positives in C1, negatives in C2 ---
        # Sim between all C1 pairs: (n1, n1)
        sim_c1_c1 = self._scaled_dot_product_sim(c1_feats, c1_feats) / self.temperature
        # Sim between C1 anchors and C2 negatives: (n1, n2)
        sim_c1_c2 = self._scaled_dot_product_sim(c1_feats, c2_feats) / self.temperature
        # Denominator: sum of exp(sim) over ALL C2 negatives
        neg_sum_c1 = torch.sum(torch.exp(sim_c1_c2), dim=1)  # (n1,)

        for i in range(n1):
            for j in range(n1):
                if i == j:
                    continue
                numerator = torch.exp(sim_c1_c1[i, j])
                loss += -torch.log(numerator / (neg_sum_c1[i] + 1e-8))
                count += 1

        # --- Anchors in C2, positives in C2, negatives in C1 ---
        if n2 >= 2:
            sim_c2_c2 = self._scaled_dot_product_sim(c2_feats, c2_feats) / self.temperature
            sim_c2_c1 = self._scaled_dot_product_sim(c2_feats, c1_feats) / self.temperature
            neg_sum_c2 = torch.sum(torch.exp(sim_c2_c1), dim=1)  # (n2,)

            for i in range(n2):
                for j in range(n2):
                    if i == j:
                        continue
                    numerator = torch.exp(sim_c2_c2[i, j])
                    loss += -torch.log(numerator / (neg_sum_c2[i] + 1e-8))
                    count += 1

        if count > 0:
            return loss / count
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
