"""The outer-loop dual controller. Implements Eq. (11) of section3-srdo.md.

    mu_{k+1} = [ mu_k + eta_mu * (tau - U_bar_k) ]_+

with the three stabilisers section 3.6 requires, because the naive update is a
random walk in practice:

  * EMA on utility     — U_bar = beta*U_bar + (1-beta)*U, killing per-batch
                         variance in the measured utility.
  * deadband           — no update while |tau - U_bar| < eps, suppressing the
                         limit cycle around an active constraint.
  * two-timescale      — the controller is stepped once per validation pass, not
                         per inner step, so theta tracks mu quasi-statically. That
                         cadence is the caller's responsibility; this class only
                         enforces eta_mu is small.

eta_mu is expressed as a fraction of mu_ref = 1/mean(Omega) (measured 11.36 in
Phase 1), so the step size transfers across datasets instead of being a raw
number that silently depends on the scale of Omega and the MSE units.

A PI variant is provided as the section-7 contingency: if pure integral control
(dual ascent) oscillates, adding a proportional term damps it. Dual ascent IS
the integral-only special case, so switching is a one-flag change, not a rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DualController:
    tau: float                    # target utility threshold
    mu_ref: float                 # 1 / mean(Omega); sets the natural scale of mu
    eta_frac: float = 0.10        # step per full violation, as a fraction of mu_ref
    beta: float = 0.9             # utility EMA
    deadband: float = 0.01        # no update inside +/- this band
    kp_frac: float = 0.0          # proportional gain (0 => pure dual ascent)
    mu_max: float | None = None   # optional clamp; unbounded mu can blow up the loss

    mu: float = 0.0
    _u_ema: float | None = None
    history: list[dict] = field(default_factory=list)

    @property
    def eta(self) -> float:
        return self.eta_frac * self.mu_ref

    @property
    def kp(self) -> float:
        return self.kp_frac * self.mu_ref

    def reset(self, mu0: float = 0.0) -> None:
        self.mu = mu0
        self._u_ema = None
        self.history.clear()

    def step(self, utility: float) -> float:
        """Consume one measured utility, update mu, return the new value.

        Call ONCE per validation pass (two-timescale separation), not per inner
        step. `utility` should be the SOFT surrogate (section 3.6), not hard
        accuracy, which is too quantised to control on.
        """
        # EMA first, so both the residual and the deadband see the smoothed value.
        self._u_ema = utility if self._u_ema is None else (
            self.beta * self._u_ema + (1.0 - self.beta) * utility
        )
        residual = self.tau - self._u_ema   # >0 when under-performing => raise mu

        prev_mu = self.mu
        if abs(residual) < self.deadband:
            step = 0.0
        else:
            # Integral term (dual ascent) + optional proportional term (PI).
            # The proportional term acts on the instantaneous residual; the
            # integral term is the accumulation embodied in mu itself.
            step = self.eta * residual + self.kp * residual

        mu_new = max(0.0, prev_mu + step)          # projection onto mu >= 0
        if self.mu_max is not None:
            mu_new = min(mu_new, self.mu_max)
        self.mu = mu_new

        self.history.append({
            "utility": utility,
            "u_ema": self._u_ema,
            "residual": residual,
            "mu": self.mu,
            "active": abs(residual) >= self.deadband,
        })
        return self.mu

    def summary(self) -> dict:
        """Convergence diagnostics for the pilot gate (section 5)."""
        if len(self.history) < 4:
            return {"n": len(self.history), "converged": None}
        recent = [h["mu"] for h in self.history[-6:]]
        span = max(recent) - min(recent)
        last_u = [h["u_ema"] for h in self.history[-4:]]
        return {
            "n": len(self.history),
            "mu": self.mu,
            "mu_span_recent": span,                       # small => settled
            "u_ema": self._u_ema,
            "at_target": abs(self.tau - (self._u_ema or 0)) < self.deadband,
            # A crude oscillation flag: mu still swinging widely late in training.
            "oscillating": span > 0.25 * max(self.mu, self.mu_ref * 0.1),
            "mean_u_recent": sum(last_u) / len(last_u),
        }
