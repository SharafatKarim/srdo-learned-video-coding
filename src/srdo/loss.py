"""The SRDO inner-loop loss. Implements Eq. (4) of section3-srdo.md.

    L_inner = mean_i[ (1 + mu * Omega_i) * D_i ] + lambda_0 * bpp

Everything the paper's contribution rests on is the (1 + mu * Omega_i) weight,
so three things must be exactly right and are asserted here rather than assumed:

  1. Omega is resampled onto whatever grid the distortion map occupies, which in
     practice is the PIXEL grid, so the pooling below is inactive by design.

     The rate term is a scalar, and that is not a defect. R_total = sum_j r_j is
     already the per-location decomposition, with r_j a function of latent j
     alone, so differentiating recovers it: stationarity in y_hat_j gives
     (1/N) sum_{i in N(j)} w_i dD_i/dy_j + lambda_0 dr_j/dy_j = 0, and dividing
     by w_j is exactly the condition for lambda_S,j = lambda_0/(1+mu*Omega_j).
     Eq. (5) is what the optimizer already solves.

     Pooling Omega to the stride-16 lattice would replace the distortion-weighted
     average of w over a latent's receptive field with an unweighted one — a
     different approximation, not a better one. Pixel resolution is kept for that
     reason; align_omega remains for callers whose distortion map is coarser.

  2. The distortion map keeps its spatial dimensions until AFTER the weight is
     applied. `F.mse_loss(..., reduction='none')` then a manual spatial mean —
     never a scalar mse that has already collapsed the map the weight needs.

  3. mu is detached from the inner graph. It is set by the outer loop (Eq. 11),
     not learned by backprop through the codec. A stray gradient into mu would
     couple the two levels and destroy the bilevel structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossTerms:
    total: torch.Tensor      # the scalar to call .backward() on
    weighted_dist: torch.Tensor
    bpp: torch.Tensor
    plain_dist: torch.Tensor   # unweighted MSE, for logging / PSNR
    semantic_dist: torch.Tensor  # S = mean(Omega * D), the constrained quantity
    mean_weight: torch.Tensor    # mean(1 + mu*Omega), sanity check


def align_omega(omega: torch.Tensor, ref_hw: tuple[int, int]) -> torch.Tensor:
    """Pool a pixel-grid Omega (B,1,H,W) down to a reference spatial size.

    Average pooling, not nearest: a latent location aggregates a block of pixels,
    so its prior should be the mean prior over that block, matching how its rate
    aggregates the block's information.
    """
    if omega.dim() == 3:
        omega = omega.unsqueeze(1)
    if omega.shape[-2:] == ref_hw:
        return omega
    return F.adaptive_avg_pool2d(omega, ref_hw)


def _flatten_tensors(obj) -> list[torch.Tensor]:
    """Collect every tensor from an arbitrarily nested dict/list/tuple.

    Scale-Space Flow returns likelihoods as a per-frame list of dicts, not a
    single dict, so the bpp accumulation must recurse rather than assume one
    flat mapping.
    """
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, dict):
        return [t for v in obj.values() for t in _flatten_tensors(v)]
    if isinstance(obj, (list, tuple)):
        return [t for v in obj for t in _flatten_tensors(v)]
    return []


def bpp_from_likelihoods(likelihoods, num_pixels: int) -> torch.Tensor:
    """Bits per pixel from an entropy model's per-element likelihoods.

    Accepts a tensor, a dict of tensors (single-frame image codecs), or a nested
    list/dict (video codecs such as SSF). num_pixels is H*W of the INPUT frame;
    the batch/time dimension is normalised by the caller.
    """
    tensors = _flatten_tensors(likelihoods)
    if not tensors:
        raise ValueError("no likelihood tensors found in codec output")
    bits = sum((-torch.log2(t.clamp_min(1e-9))).sum() for t in tensors)
    return bits / num_pixels


def srdo_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    likelihoods,
    omega: torch.Tensor,
    mu: float | torch.Tensor,
    lambda_0: float,
    num_pixels: int | None = None,
    normalize_weight: bool = False,
) -> LossTerms:
    """Compute the SRDO inner loss.

    x_hat, x : (B, C, H, W) reconstructions and targets in [0, 1].
    omega    : (B, 1, H, W) or (B, H, W) semantic prior in [0, 1].
    mu       : scalar dual variable, HELD FIXED this step (Eq. 11 sets it).
    lambda_0 : base RD multiplier, fixing the operating point.
    normalize_weight : divide the weight by its own mean, so that raising mu
        changes only the SPATIAL DISTRIBUTION of distortion and not the total.

    On normalize_weight. The unnormalised weight has mean 1 + mu*E[Omega], so
    with E[Omega] ~ 0.09 a sweep to mu=60 scales the whole distortion term by
    6.3x, which is arithmetically the same as dividing lambda_0 by 6.3. The
    measured consequence is that a mu sweep traverses rate as well as
    allocation: bpp rose 2.4x and PLAIN distortion improved 2.1x across the
    sweep, which is what a simple rate increase produces and not what
    reallocation predicts. The genuine spatial effect over that sweep is an 18%
    fall in S/D, visible only once the rate move is accounted for.

    Normalising corresponds to the two-constraint Lagrangian, where a bound on
    total distortion carries its own multiplier: minimising R subject to both
    sum(D_i) <= D_max and sum(Omega_i D_i) <= S_max gives
    R + lambda*sum((1 + (nu/lambda) Omega_i) D_i), in which lambda moves to hold
    the first constraint while mu = nu/lambda sets only the split. Folding both
    into a single fixed lambda_0, as the unnormalised form does, leaves nothing
    holding the operating point still.

    Defaults to False so that runs already collected remain reproducible; the
    normalised form is what a rate-conserving sweep should use.
    """
    if isinstance(mu, torch.Tensor):
        mu = mu.detach()  # (2) never backprop into mu

    b, c, h, w = x.shape
    num_pixels = num_pixels if num_pixels is not None else h * w

    # (2) per-location distortion, spatial dims intact, channels averaged
    dist_map = F.mse_loss(x_hat, x, reduction="none").mean(dim=1, keepdim=True)  # (B,1,H,W)

    # (1) align the prior to the distortion map's grid
    omega_a = align_omega(omega, dist_map.shape[-2:]).clamp(0.0, 1.0)

    weight = 1.0 + mu * omega_a                        # (1 + mu*Omega_i)
    if normalize_weight:
        # Per batch, not per corpus: E[Omega] varies enough between clips that a
        # global constant would leave the operating point drifting clip to clip.
        weight = weight / weight.mean().clamp_min(1e-8)
    weighted_dist = (weight * dist_map).mean()
    bpp = bpp_from_likelihoods(likelihoods, num_pixels) / b

    total = weighted_dist + lambda_0 * bpp

    return LossTerms(
        total=total,
        weighted_dist=weighted_dist.detach(),
        bpp=bpp.detach(),
        plain_dist=dist_map.mean().detach(),
        semantic_dist=(omega_a * dist_map).mean().detach(),
        mean_weight=weight.mean().detach(),
    )
