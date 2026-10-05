"""Kaggle GPU Training Script: 5-Fold Leave-One-Scene-Out Cross-Validation & Open Legs.

Runs true 5-fold cross-validation across all 5 scenes in Domain D:
  Fold 0: val_scene = "scene_S001"
  Fold 1: val_scene = "scene_S003"
  Fold 2: val_scene = "scene_S004"
  Fold 3: val_scene = "scene_S008"
  Fold 4: val_scene = "scene_S009"

Also includes extended step legs (4000 steps) for high-utility lambda_0=0.002.
Saves checkpoints as `codec_fold_{scene}_mu{mu}_lam{lam}.pt` into /kaggle/working/.
"""

import pathlib
import sys
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from compressai.zoo import ssf2020
from tqdm.auto import tqdm

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
from srdo.evaluate import evaluate_codec, save_checkpoint, ssf_forward  # noqa: E402
from srdo.loss import srdo_loss  # noqa: E402

DATA_ROOT = dd.find_dataset_root()
print(f"data:    {DATA_ROOT}")

SCENES = ["scene_S001", "scene_S003", "scene_S004", "scene_S008", "scene_S009"]
LAMBDAS = [0.002, 0.005, 0.01, 0.03, 0.08]
MUS = [0.0, 60.0]

CFG = dict(
    patch=dd.PATCH,
    lr=1e-4,
    eval_every=500,
    seed=0,
    normalize_weight=True,
    val_windows_per_clip=8,
)
_rng = np.random.default_rng(CFG["seed"])


def train_one_fold(val_scene, mu, lam, steps, store, tr_clips, va, accelerator):
    codec = ssf2020(quality=4, pretrained=True).train()
    codec.update(force=True)
    opt = torch.optim.Adam((p for p in codec.parameters() if p.requires_grad), lr=CFG["lr"])
    codec, opt = accelerator.prepare(codec, opt)

    def load_val(row):
        return dd.make_gop(store, row["_clip"], int(row["_idx"]))

    label = f"fold_{val_scene}_mu{mu:g}_lam{lam:g}"
    curve = []
    pbar = tqdm(range(1, steps + 1), desc=label, disable=not accelerator.is_local_main_process)
    
    for step in pbar:
        clip = tr_clips[int(_rng.integers(len(tr_clips)))]
        x, om = dd.make_gop(store, clip, int(_rng.integers(store[clip].shape[0])))
        x, om = x.to(accelerator.device), om.to(accelerator.device)

        x_hat, lik = ssf_forward(codec, x)
        terms = srdo_loss(x_hat, x, lik, om, mu=mu, lambda_0=lam,
                          num_pixels=CFG["patch"] ** 2,
                          normalize_weight=CFG["normalize_weight"])
        opt.zero_grad()
        accelerator.backward(terms.total)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(codec.parameters(), 1.0)
        opt.step()

        if accelerator.is_local_main_process:
            pbar.set_postfix(bpp=f"{float(terms.bpp):.4f}")

        if step % CFG["eval_every"] == 0:
            ev = evaluate_codec(accelerator.unwrap_model(codec),
                                (r for _, r in va.iterrows()), load_val,
                                device=accelerator.device, num_pixels=CFG["patch"] ** 2)
            if accelerator.is_local_main_process:
                row = {"val_scene": val_scene, "mu": mu, "lambda_0": lam, "step": step, **ev.summary(tau=0.45)}
                curve.append(row)
                ev.per_clip.to_csv(WORK / f"perclip_{label}_step{step}.csv", index=False)

    if accelerator.is_local_main_process:
        save_checkpoint(WORK / f"codec_{label}.pt", accelerator.unwrap_model(codec),
                        mu=mu, step=steps, lambda_0=lam)
    return curve


def main():
    accelerator = Accelerator(
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])

    store = dd.preload(DATA_ROOT)
    all_rows = []

    for val_scene in SCENES:
        tr_clips, val_clips = dd.split_by_scene(store, val_scene)
        va = dd.build_val(store, val_clips, CFG["val_windows_per_clip"])
        print(f"\n================ Fold: {val_scene} ================")
        print(f"train {len(tr_clips)} clips; val {len(val_clips)} clips -> {len(va)} GOPs\n")

        for mu in MUS:
            for lam in LAMBDAS:
                # Extend train steps to 4000 for lam=0.002 leg settling check
                steps = 4000 if lam == 0.002 else 2000
                rows = train_one_fold(val_scene, mu, lam, steps, store, tr_clips, va, accelerator)
                all_rows += rows
                if accelerator.is_local_main_process:
                    df = pd.DataFrame(all_rows)
                    df.to_csv(WORK / "true_5fold_cv_results.csv", index=False)
                torch.cuda.empty_cache()

    print("\nAll 5-fold cross-validation training runs complete!")


if __name__ == "__main__":
    main()
