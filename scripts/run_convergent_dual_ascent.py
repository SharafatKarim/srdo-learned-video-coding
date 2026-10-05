"""Convergent Dual-Ascent Simulation and Controller Test.

Tests the DualController (src/srdo/dual.py) initialized at tau = 0.354 (the 80% knee),
eta_frac = 0.44 (eta_mu = 5.0), beta = 0.9, deadband = 0.01 against the measured plant
model derived from the domain D neural sweep.

Saves summary and trajectory outputs to outputs/phase4_dual_convergent_simulation.csv.
"""

from __future__ import annotations

import pathlib
import sys
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from srdo.dual import DualController
from srdo.bdrate import RDCurve

OUT_CSV = ROOT / "outputs" / "phase4_dual_convergent_simulation.csv"


def simulate_dual_ascent(tau=0.354, eta_frac=0.44, mu_ref=11.36, beta=0.9, 
                         deadband=0.01, mu0=5.0, steps=25, seed=42):
    """Simulates outer-loop dual ascent updates against the empirical plant model."""
    np.random.seed(seed)
    
    # Empirical plant response U(mu) fitted from domain D / consumer IP sweeps
    # U_sweep(mu) = 0.309 + 0.027 * (1 - exp(-mu / 15.0))
    def plant_u(mu):
        base_u = 0.309 + 0.027 * (1.0 - np.exp(-mu / 15.0))
        noise = np.random.normal(0, 0.0076)
        return base_u + noise

    controller = DualController(
        tau=tau,
        mu_ref=mu_ref,
        eta_frac=eta_frac,
        beta=beta,
        deadband=deadband,
        kp_frac=0.0,
        mu_max=60.0
    )
    controller.reset(mu0=mu0)

    records = []
    current_mu = mu0
    
    for k in range(steps):
        u_meas = plant_u(current_mu)
        new_mu = controller.step(u_meas)
        
        rec = controller.history[-1]
        rec["step"] = k
        rec["measured_u"] = u_meas
        records.append(rec)
        current_mu = new_mu

    df = pd.DataFrame(records)
    summary = controller.summary()
    return df, summary


def main():
    print("=== Convergent Dual-Ascent Simulation (tau=0.354, 80% Knee) ===")
    
    df_sim, summary = simulate_dual_ascent(
        tau=0.354,
        eta_frac=0.44,
        mu_ref=11.36,
        beta=0.9,
        deadband=0.01,
        mu0=5.0,
        steps=30,
        seed=42
    )
    
    df_sim.to_csv(OUT_CSV, index=False)
    print(f"\nSaved trajectory records to {OUT_CSV}")
    print(f"\nConvergence Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    
    print("\nTrajectory snippet (last 10 steps):")
    print(df_sim[["step", "measured_u", "u_ema", "residual", "mu", "active"]].tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
