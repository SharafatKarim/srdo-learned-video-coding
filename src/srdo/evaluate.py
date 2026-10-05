"""Validation-set evaluation for the phase 4 codec runs.

The pilot notebooks evaluated with a function that returned three means and
nothing else. Four things the paper needs were destroyed by that reduction:

  * PER-CLIP UTILITY. Section 5 reports a constraint-satisfaction rate — the
    fraction of clips meeting tau — which a mean cannot produce. A mean of 0.52
    is consistent with every clip at 0.52 and with half the corpus at 0.9 and
    half at 0.14, and those are different codecs.

  * S ON THE VALIDATION SET. The sweep logged S from a single random training
    GOP at the last step, which is why the logged S is non-monotone in mu and
    smallest at mu=0, where it must be largest. S is the constrained quantity;
    it has to be measured where U is measured, on the same reconstructions.

  * SKIP ACCOUNTING. Clips whose clean pass yields no detections are dropped,
    since pseudo-GT is undefined for them. That is correct, but the count must
    be reported: a val set that silently shrinks to a third of its size makes
    every mean noisier than the plot suggests.

  * DISPERSION. Without a standard error, a 0.02 difference between two runs
    cannot be told from noise, and several of the pilot's decisions turned on
    differences that size.

Nothing here is codec-specific. `forward` and `load` are injected so the same
function serves the image-codec and Scale-Space-Flow paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .detection import detect, evaluate as evaluate_detections
from .loss import align_omega, bpp_from_likelihoods


def frames_to_uint8(x: torch.Tensor) -> np.ndarray:
    """(T,C,H,W) float in [0,1] -> (T,H,W,3) uint8 RGB for the detector."""
    return (x.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8")


def ssf_forward(codec, x: torch.Tensor):
    """Scale-Space Flow takes a list of frames and returns a list of x_hat."""
    out = codec([f.unsqueeze(0) for f in x])
    x_hat = out["x_hat"]
    x_hat = torch.cat(x_hat, 0) if isinstance(x_hat, list) else x_hat
    return x_hat, out["likelihoods"]


@dataclass(frozen=True)
class ValEval:
    """Validation result. `per_clip` is the primary object; means are derived."""

    per_clip: pd.DataFrame
    n_requested: int
    n_scored: int
    n_skipped_no_gt: int
    n_failed: int

    @property
    def soft_u(self) -> float:
        return self._mean("soft_u")

    @property
    def hard_u(self) -> float:
        return self._mean("hard_u")

    @property
    def bpp(self) -> float:
        return self._mean("bpp")

    @property
    def s_semantic(self) -> float:
        """S = mean_i(Omega_i * D_i), averaged over clips. The constrained quantity."""
        return self._mean("s_semantic")

    @property
    def d_plain(self) -> float:
        return self._mean("d_plain")

    def _mean(self, col: str) -> float:
        if self.per_clip.empty or col not in self.per_clip:
            return float("nan")
        return float(self.per_clip[col].mean())

    def sem(self, col: str = "soft_u") -> float:
        """Standard error of the clip mean. Two runs closer than ~2 sem are tied."""
        n = len(self.per_clip)
        if n < 2:
            return float("nan")
        return float(self.per_clip[col].std(ddof=1) / np.sqrt(n))

    def satisfaction_rate(self, tau: float, col: str = "soft_u") -> float:
        """Fraction of clips at or above tau — the section 5 primary result.

        The constraint is written per-clip in section 3, so this, not the mean,
        is what 'the constraint is satisfied' means.
        """
        if self.per_clip.empty:
            return float("nan")
        return float((self.per_clip[col] >= tau).mean())

    def summary(self, tau: float | None = None) -> dict:
        out = {
            "soft_u": self.soft_u, "soft_u_sem": self.sem("soft_u"),
            "hard_u": self.hard_u, "bpp": self.bpp,
            "s_semantic": self.s_semantic, "d_plain": self.d_plain,
            "n_scored": self.n_scored, "n_skipped_no_gt": self.n_skipped_no_gt,
            "n_failed": self.n_failed,
        }
        if tau is not None:
            out["satisfaction_rate"] = self.satisfaction_rate(tau)
        return out


@torch.no_grad()
def evaluate_codec(
    codec,
    rows: Iterable,
    load: Callable[[object], tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device | str = "cpu",
    num_pixels: int | None = None,
    forward: Callable = ssf_forward,
    key_fields: tuple[str, ...] = ("clip", "action", "source"),
) -> ValEval:
    """Score a codec clip by clip.

    codec : nn.Module. Left in whatever mode it arrived in.
    rows  : iterable of manifest rows (anything with __getitem__ by column name).
    load  : row -> (x, omega), x (T,C,H,W) in [0,1], omega (T,1,H,W) in [0,1].
    """
    was_training = codec.training
    codec.eval()

    records: list[dict] = []
    n_requested = n_skipped = n_failed = 0

    for row in rows:
        n_requested += 1
        try:
            x, omega = load(row)
            x = x.to(device)
            omega = omega.to(device)

            # Pseudo-GT from the CLEAN clip. Undefined if the detector fires on
            # nothing, so those clips are excluded rather than scored as zero —
            # scoring them zero would punish the codec for the detector's miss.
            ref = detect(frames_to_uint8(x))
            if ref.n_total == 0:
                n_skipped += 1
                continue

            x_hat, likelihoods = forward(codec, x)
            npx = num_pixels if num_pixels is not None else x.shape[-2] * x.shape[-1]
            bpp = float(bpp_from_likelihoods(likelihoods, npx) / x.shape[0])

            hard_u, soft_u = evaluate_detections(ref, detect(frames_to_uint8(x_hat)))
            if soft_u != soft_u:  # NaN
                n_skipped += 1
                continue

            # S and D from the SAME reconstruction that produced soft_u, so the
            # constraint and the utility refer to one codec state, not two.
            dist_map = F.mse_loss(x_hat, x, reduction="none").mean(dim=1, keepdim=True)
            omega_a = align_omega(omega, dist_map.shape[-2:]).clamp(0.0, 1.0)

            rec = {f: row[f] for f in key_fields if f in getattr(row, "index", [])}
            rec.update({
                "soft_u": float(soft_u),
                "hard_u": float(hard_u),
                "bpp": bpp,
                "s_semantic": float((omega_a * dist_map).mean()),
                "d_plain": float(dist_map.mean()),
                "mean_omega": float(omega_a.mean()),
                "n_gt": int(ref.n_total),
            })
            records.append(rec)

        except Exception as exc:  # one bad clip must not lose the whole pass
            n_failed += 1
            print(f"  eval failed on {getattr(row, 'get', lambda *_: '?')('clip', '?')}: "
                  f"{type(exc).__name__}: {exc}")

    if was_training:
        codec.train()

    return ValEval(
        per_clip=pd.DataFrame.from_records(records),
        n_requested=n_requested,
        n_scored=len(records),
        n_skipped_no_gt=n_skipped,
        n_failed=n_failed,
    )


def save_checkpoint(path, codec, *, mu: float, step: int, lambda_0: float,
                    extra: dict | None = None) -> None:
    """Persist enough to resume or warm-start.

    The fixed-mu sweep deleted each codec after evaluating it, so no run can be
    warm-started from a converged operating point and every dual run had to
    begin from a randomly-adapted mu=0 codec. Storing mu and lambda_0 alongside
    the weights is what makes a checkpoint identifiable later.
    """
    state = codec.module.state_dict() if hasattr(codec, "module") else codec.state_dict()
    torch.save({
        "state_dict": state,
        "mu": float(mu),
        "step": int(step),
        "lambda_0": float(lambda_0),
        **(extra or {}),
    }, str(path))
