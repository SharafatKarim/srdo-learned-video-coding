"""Kaggle GPU Script: Continuous Online Dual-Ascent & High-mu0 Convergence-from-Above.

Executes real-time online dual-ascent fine-tuning of Scale-Space Flow.
Updates model weights live at every outer step rather than using a discrete lookup table.

Runs two trajectories:
1. Standard warm start: mu0 = 5.0 (Approach from below)
2. High initialisation: mu0 = 30.0 (Approach from above / convergence test)
"""

import pathlib
import sys
import numpy as np
import pandas as pd
import torch
from compressai.zoo import ssf2020

_MODEL_BASE = "/kaggle/input/models/msharafatkarim/python-cctv-srdo/pytorch/default"
WORK = pathlib.Path("/kaggle/working")


def _latest_package(base=_MODEL_BASE):
    versions = sorted(int(p.name) for p in pathlib.Path(base).iterdir() if p.name.isdigit())
    if not versions:
        raise RuntimeError(f"no numbered model version attached under {base}")
    return f"{base}/{versions[-1]}"


SRDO_SRC = _latest_package()
print(f"package: {SRDO_SRC}")
sys.path.insert(0, SRDO_SRC)

from srdo import domain_d as dd  # noqa: E402
from srdo.detection import detect, evaluate as evaluate_detections  # noqa: E402
from srdo.dual import DualController  # noqa: E402
from srdo.evaluate import ssf_forward, frames_to_uint8  # noqa: E402
from srdo.loss import srdo_loss  # noqa: E402

DATA_ROOT = dd.find_dataset_root()
VAL_SCENE = "scene_S009"


def run_continuous_dual_ascent(mu0: float, name_tag: str):
    print(f"\n=== Starting Live Continuous Dual Ascent Run (mu0={mu0}) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = dd.preload(DATA_ROOT)
    tr_clips, val_clips = dd.split_by_scene(store, VAL_SCENE)
    va = dd.build_val(store, val_clips, 8)

    # Initialize model
    codec = ssf2020(quality=4, pretrained=True).to(device)
    codec.train()
    opt = torch.optim.Adam((p for p in codec.parameters() if p.requires_grad), lr=1e-4)

    controller = DualController(
        tau=0.354,
        mu_ref=11.36,
        eta_frac=0.44,   # eta_mu = 5.0
        beta=0.9,
        deadband=0.01,
        kp_frac=0.0,
        mu_max=60.0
    )
    controller.reset(mu0=mu0)
    current_mu = mu0
    lam_0 = 0.01
    _rng = np.random.default_rng(42)

    records = []
    outer_steps = 25
    inner_steps_per_outer = 500  # Fine-tune codec for 500 steps per outer update

    for step_out in range(1, outer_steps + 1):
        # 1. Inner Loop: Fine-tune neural codec weights at current_mu
        codec.train()
        for _ in range(inner_steps_per_outer):
            clip = tr_clips[int(_rng.integers(len(tr_clips)))]
            x, om = dd.make_gop(store, clip, int(_rng.integers(store[clip].shape[0])))
            x, om = x.to(device), om.to(device)

            x_hat, lik = ssf_forward(codec, x)
            terms = srdo_loss(x_hat, x, lik, om, mu=current_mu, lambda_0=lam_0,
                              num_pixels=dd.PATCH ** 2, normalize_weight=True)
            opt.zero_grad()
            terms.total.backward()
            torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
            opt.step()

        # 2. Outer Loop: Evaluate on held-out validation set and update mu
        codec.eval()
        soft_u_list, hard_u_list, bpp_list = [], [], []

        for _, row in va.iterrows():
            gop = dd.make_gop(store, row["_clip"], int(row["_idx"]))[0]
            t_len, h, w, _ = gop.shape
            x_clean = torch.from_numpy(gop).permute(0, 3, 1, 2).float() / 255.0
            det_clean = detect(gop)
            if sum(len(b) for b in det_clean.boxes) == 0:
                continue

            with torch.no_grad():
                x_hat, likelihoods = ssf_forward(codec, x_clean.to(device))

            rec_uint8 = frames_to_uint8(x_hat.cpu())
            det_rec = detect(rec_uint8)
            hard_u, soft_u = evaluate_detections(det_clean, det_rec)

            soft_u_list.append(soft_u)
            hard_u_list.append(hard_u)

        soft_u_mean = float(np.mean(soft_u_list)) if soft_u_list else 0.0
        hard_u_mean = float(np.mean(hard_u_list)) if hard_u_list else 0.0

        new_mu = controller.step(soft_u_mean)
        hist = controller.history[-1]
        hist.update({
            "outer_step": step_out,
            "hard_u_f1": hard_u_mean,
            "mu0_start": mu0
        })
        records.append(hist)

        print(f"[{name_tag}] Step {step_out:2d} | mu = {current_mu:6.2f} -> {new_mu:6.2f} | "
              f"soft_U = {soft_u_mean:.4f} | hard_F1 = {hard_u_mean:.4f} | active = {hist['active']}")
        current_mu = new_mu

    df_out = pd.DataFrame(records)
    df_out.to_csv(WORK / f"continuous_dual_ascent_{name_tag}.csv", index=False)
    print(f"Saved {name_tag} trajectory to {WORK / f'continuous_dual_ascent_{name_tag}.csv'}")


def main():
    # 1. Standard warm-start trajectory from below (mu0 = 5.0)
    run_continuous_dual_ascent(mu0=5.0, name_tag="mu0_5")

    # 2. High initialisation trajectory from above (mu0 = 30.0)
    run_continuous_dual_ascent(mu0=30.0, name_tag="mu0_30")


if __name__ == "__main__":
    main()
