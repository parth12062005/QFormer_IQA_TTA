"""
TTA Stabilization Strategies for the Projection Head.

Four strategies to improve TTA stability:

1. Identity-ish Initialization:
       W ≈ I so that z = Wf + b ≈ f[:out_dim]
   Preserves source-trained feature geometry from the very first TTA step.

2. Projection Head Pretraining:
   Pre-adapt the projection head on the test loader using auxiliary losses
   *before* any backbone adaptation begins.

3. Staged TTA (Warmup):
   Within each batch, first N steps train ONLY the projection head
   (backbone frozen), then unfreeze backbone for remaining steps.

4. EMA Stabilization:
   θ_ema = α·θ_ema + (1−α)·θ   (α ≈ 0.99)
   Provides a temporally smoothed projection used as a stability anchor.
"""

import copy
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Identity-ish Initialization
# ---------------------------------------------------------------------------
def init_proj_head_identity(proj_head):
    """
    Initialize projection head weights as a truncated identity matrix.

    For Linear(in_dim, out_dim) where out_dim <= in_dim, W is set to the
    first *out_dim* rows of I_{in_dim}, bias to zero.  This ensures
    z ≈ f[:out_dim] at initialization, preserving the backbone feature
    geometry that was learned during source training.
    """
    for module in proj_head.modules():
        if isinstance(module, nn.Linear):
            in_dim = module.in_features
            out_dim = module.out_features
            min_dim = min(in_dim, out_dim)
            with torch.no_grad():
                nn.init.zeros_(module.weight)
                module.weight[:min_dim, :min_dim].copy_(torch.eye(min_dim))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


def init_proj_head_orthogonal(proj_head):
    """
    Initialize projection head with orthogonal weights (norm-preserving).
    """
    for module in proj_head.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
# JL-Lemma-Compliant Random Projection Initializations
#
# For a linear projection U ∈ R^{p × d} mapping d-dimensional features into
# p dimensions, the Johnson-Lindenstrauss lemma requires that U_ij are i.i.d.
# with:
#     μ = 0        (zero mean — cross terms cancel in expectation)
#     σ² = 1/p     (scaled unit variance — so summing p dimensions
#                   recovers the original squared distance)
#
# Any distribution satisfying these two constraints is a valid JL projection.
# Three families are implemented below.
# ---------------------------------------------------------------------------

def init_proj_head_jl_gaussian(proj_head):
    """
    JL Gaussian projection:  U_ij ~ N(0, 1/p).

    The classic choice from the JL-lemma proofs.  Each element is drawn from
    a zero-mean Gaussian whose variance is inversely proportional to the
    target dimension p (= out_features), so:
        E[||Ux||²] = ||x||²   for any vector x.
    """
    for module in proj_head.modules():
        if isinstance(module, nn.Linear):
            p = module.out_features
            with torch.no_grad():
                module.weight.normal_(mean=0.0, std=(1.0 / p) ** 0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


def init_proj_head_jl_sparse(proj_head):
    """
    JL Sparse (Achlioptas) projection:

        U_ij = { +√(3/p)  with prob 1/6
               {  0       with prob 2/3
               { -√(3/p)  with prob 1/6

    From Achlioptas (2003, "Database-friendly random projections").
    E[U_ij] = 0,  Var[U_ij] = (3/p)·(1/3) = 1/p.

    ~67% of the matrix is zero, so the projection is a sparse matrix
    multiply — mostly additions and subtractions, very fast.
    """
    for module in proj_head.modules():
        if isinstance(module, nn.Linear):
            p = module.out_features
            val = (3.0 / p) ** 0.5
            with torch.no_grad():
                # Draw uniform [0,1) and map to the 3-point distribution
                u = torch.rand_like(module.weight)
                w = torch.zeros_like(module.weight)
                w[u < 1.0 / 6.0] = val         # prob 1/6 → +val
                w[u > 5.0 / 6.0] = -val        # prob 1/6 → -val
                # middle 2/3 stays zero
                module.weight.copy_(w)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


def init_proj_head_jl_rademacher(proj_head):
    """
    JL Rademacher (sign-flip) projection:

        U_ij = ±1/√p   each with probability 1/2.

    E[U_ij] = 0,  Var[U_ij] = 1/p.

    Dense but extremely cheap: every multiply is just ±1 scaled by a
    constant, i.e. additions and subtractions only (no real multiplications).
    """
    for module in proj_head.modules():
        if isinstance(module, nn.Linear):
            p = module.out_features
            scale = (1.0 / p) ** 0.5
            with torch.no_grad():
                signs = torch.randint(0, 2, module.weight.shape,
                                      device=module.weight.device,
                                      dtype=module.weight.dtype) * 2 - 1
                module.weight.copy_(signs * scale)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


PROJ_INIT_REGISTRY = {
    "identity":       init_proj_head_identity,
    "orthogonal":     init_proj_head_orthogonal,
    "jl_gaussian":    init_proj_head_jl_gaussian,
    "jl_sparse":      init_proj_head_jl_sparse,
    "jl_rademacher":  init_proj_head_jl_rademacher,
}


# ---------------------------------------------------------------------------
# 4. EMA Stabilization
# ---------------------------------------------------------------------------
class EMAProjectionHead:
    """
    Exponential Moving Average wrapper for the projection head.

    Maintains a slowly-updated teacher copy whose parameters evolve as:
        θ_ema = α·θ_ema + (1−α)·θ_student

    The EMA projections are added to the loss context as ``ema_proj_feats``
    so that losses can optionally use them as a stability reference.
    """

    def __init__(self, proj_head: nn.Module, decay: float = 0.99):
        self.decay = decay
        self.ema_head = copy.deepcopy(proj_head)
        self.ema_head.eval()
        for p in self.ema_head.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, proj_head: nn.Module):
        """Update EMA parameters from the student."""
        for ema_p, p in zip(self.ema_head.parameters(), proj_head.parameters()):
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the EMA projection head (no grad)."""
        return self.ema_head(x)

    def reset_from(self, proj_head: nn.Module):
        """Re-sync the EMA to match the student (used after per-batch reset)."""
        for ema_p, p in zip(self.ema_head.parameters(), proj_head.parameters()):
            ema_p.data.copy_(p.data)
