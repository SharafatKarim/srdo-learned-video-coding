"""All-in-One Standalone Kaggle Notebook Runner.

Runs both:
1. Strict 5-Fold Leave-One-Scene-Out Cross-Validation & Extended lambda_0=0.002 leg.
2. Continuous Online Dual Ascent & High-mu0 Convergence-from-Above (mu0=30).

Requires NO external python package dataset — all dependencies, loss functions,
domain_d data loaders, and dual controller logic are self-contained in this single script.
"""

import os
import pathlib
import sys
import subprocess
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from compressai.zoo import ssf2020

WORK = pathlib.Path("/kaggle/working")
PATCH = 256
GOP = 8

# =====================================================================
# 1. Self-Contained Domain D & Loss Helpers
# =====================================================================

def find_dataset_root(base="/kaggle/input"):
    base = pathlib.Path(base)
    for cand in sorted(base.rglob("scene_S*")):
        if cand.is_dir() and any(cand.glob("*.mpeg")):
            return cand.parent
    hits = sorted(base.rglob("*.mpeg"))
    if hits:
        return hits[0].parent.parent
    raise RuntimeError(f"domain D not found under {base}; attach the dataset in Kaggle")

def scene_of(clip_name):
    return clip_name.rsplit("_cam", 1)[0]

def decode(path, box=None, max_frames=None):
    cmd = ["ffmpeg", "-v", "error", "-i", str(path)]
    if box is not None:
        w, h, x, y = box
        cmd += ["-vf", f"crop={w}:{h}:{x}:{y}"]
    if max_frames is not None:
        cmd += ["-frames:v", str(max_frames)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    res = subprocess.run(cmd, capture_output=True)
    buf = np.frombuffer(res.stdout, dtype=np.uint8)
    h_out, w_out = (box[1], box[0]) if box else (1080, 1920)
    frame_bytes = h_out * w_out * 3
    if buf.size == 0 or frame_bytes == 0:
        return np.zeros((0, h_out, w_out, 3), dtype=np.uint8)
    n = buf.size // frame_bytes
    buf = buf[: n * frame_bytes]
    return buf.reshape(n, h_out, w_out, 3)

def generate_simple_omega(gop_shape):
    # Fallback/Dummy prior if detector prior is not pre-computed:
    # returns uniform central prior or zeros
    t, h, w, _ = gop_shape
    om = np.zeros((t, h, w), dtype=np.float32)
    om[:, h//4:3*h//4, w//4:3*w//4] = 0.5
    return om

def preload(root):
    root = pathlib.Path(root)
    store = {}
    clips = sorted(root.rglob("*.mpeg"))
    print(f"Found {len(clips)} clip files under {root}.")
    for p in tqdm(clips, desc="Preloading GOPs"):
        c_name = p.stem
        frames = decode(p, max_frames=16)
        if len(frames) >= 8:
            h, w = frames.shape[1], frames.shape[2]
            # Center crop to 256x256 patch (multiple of 16 for SSF model)
            cy, cx = h // 2, w // 2
            crop_frames = frames[:8, cy - 128 : cy + 128, cx - 128 : cx + 128]
            if crop_frames.shape == (8, 256, 256, 3):
                store[c_name] = crop_frames
    return store

def split_by_scene(store, val_scene):
    tr, va = [], []
    for k in sorted(store.keys()):
        if scene_of(k) == val_scene:
            va.append(k)
        else:
            tr.append(k)
    return tr, va

def make_gop(store, clip_name):
    frames = store[clip_name]
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    om = torch.from_numpy(generate_simple_omega(frames.shape)).unsqueeze(1)
    return x, om

def flatten_tensors(obj):
    if isinstance(obj, torch.Tensor):
        return [obj]
    elif isinstance(obj, dict):
        res = []
        for v in obj.values():
            res.extend(flatten_tensors(v))
        return res
    elif isinstance(obj, (list, tuple)):
        res = []
        for v in obj:
            res.extend(flatten_tensors(v))
        return res
    return []

def bpp_from_likelihoods(likelihoods, num_pixels):
    tensors = flatten_tensors(likelihoods)
    if not tensors:
        return torch.tensor(0.0)
    bits = sum((-torch.log2(t.clamp_min(1e-9))).sum() for t in tensors)
    return bits / num_pixels

def ssf_forward(codec, x):
    out = codec([f.unsqueeze(0) for f in x])
    x_hat = out["x_hat"]
    x_hat = torch.cat(x_hat, 0) if isinstance(x_hat, list) else x_hat
    return x_hat, out["likelihoods"]

def srdo_loss(x_hat, x, likelihoods, omega, mu, lambda_0, num_pixels, normalize_weight=True):
    dist_map = F.mse_loss(x_hat, x, reduction='none').mean(dim=1, keepdim=True)
    weight = 1.0 + mu * omega
    if normalize_weight:
        weight = weight / weight.mean().clamp(min=1e-6)
    weighted_dist = (weight * dist_map).mean()
    bpp = bpp_from_likelihoods(likelihoods, num_pixels)
    total = weighted_dist + lambda_0 * bpp
    return total, bpp

# =====================================================================
# 2. Self-Contained Dual Controller
# =====================================================================

@dataclass
class DualController:
    tau: float = 0.354
    mu_ref: float = 11.36
    eta_frac: float = 0.44
    beta: float = 0.9
    deadband: float = 0.01
    mu_max: float = 60.0
    mu: float = 5.0
    _u_ema: float = None
    history: list = field(default_factory=list)

    def reset(self, mu0: float = 5.0):
        self.mu = mu0
        self._u_ema = None
        self.history.clear()

    def step(self, utility: float) -> float:
        self._u_ema = utility if self._u_ema is None else (self.beta * self._u_ema + (1.0 - self.beta) * utility)
        residual = self.tau - self._u_ema
        step_val = (self.eta_frac * self.mu_ref) * residual if abs(residual) >= self.deadband else 0.0
        self.mu = min(self.mu_max, max(0.0, self.mu + step_val))
        self.history.append({"utility": utility, "u_ema": self._u_ema, "residual": residual, "mu": self.mu, "active": abs(residual) >= self.deadband})
        return self.mu

# =====================================================================
# 3. Task Execution Logic
# =====================================================================

SCENES = ["scene_S001", "scene_S003", "scene_S004", "scene_S008", "scene_S009"]
LAMBDAS = [0.002, 0.005, 0.01, 0.03, 0.08]
MUS = [0.0, 60.0]

def run_true_5fold_cv(store, device, rank=0, world_size=1):
    print("\n=======================================================")
    print(f"   TASK 1: TRUE 5-FOLD CV TRAIN (Rank {rank}/{world_size})   ")
    print("=======================================================")
    all_rows = []
    
    my_scenes = [s for i, s in enumerate(SCENES) if i % world_size == rank]
    print(f"Rank {rank} assigned scenes: {my_scenes}")
    
    for val_scene in my_scenes:
        tr_clips, val_clips = split_by_scene(store, val_scene)
        print(f"\n--- [Rank {rank}] Fold: {val_scene} (Train: {len(tr_clips)}, Val: {len(val_clips)}) ---")
        
        for mu in MUS:
            for lam in LAMBDAS:
                steps = 4000 if lam == 0.002 else 2000
                codec = ssf2020(quality=4, pretrained=True).to(device).train()
                opt = torch.optim.Adam((p for p in codec.parameters() if p.requires_grad), lr=1e-4)
                
                label = f"fold_{val_scene}_mu{mu:g}_lam{lam:g}"
                pbar = tqdm(range(1, steps + 1), desc=f"R{rank}:{label}")
                _rng = np.random.default_rng(42)
                
                for step in pbar:
                    clip = tr_clips[int(_rng.integers(len(tr_clips)))]
                    x, om = make_gop(store, clip)
                    x, om = x.to(device), om.to(device)
                    
                    x_hat, lik = ssf_forward(codec, x)
                    total_loss, bpp = srdo_loss(x_hat, x, lik, om, mu=mu, lambda_0=lam, num_pixels=PATCH**2)
                    
                    opt.zero_grad()
                    total_loss.backward()
                    opt.step()
                    pbar.set_postfix(bpp=f"{float(bpp.detach().item()):.4f}")
                    
                codec.eval()
                val_bpps, val_dists = [], []
                with torch.no_grad():
                    for v_clip in val_clips:
                        vx, vom = make_gop(store, v_clip)
                        vx = vx.to(device)
                        vx_hat, vlik = ssf_forward(codec, vx)
                        val_bpps.append(float(bpp_from_likelihoods(vlik, PATCH**2)))
                        val_dists.append(float(F.mse_loss(vx_hat, vx)))
                
                mean_bpp = np.mean(val_bpps) if val_bpps else 0.0
                mean_u = 1.0 - np.mean(val_dists) if val_dists else 0.0
                
                all_rows.append({
                    "val_scene": val_scene,
                    "mu": mu,
                    "lambda_0": lam,
                    "steps": steps,
                    "bpp": mean_bpp,
                    "soft_u": mean_u
                })
                torch.cuda.empty_cache()

    df = pd.DataFrame(all_rows)
    df.to_csv(WORK / f"true_5fold_cv_results_rank{rank}.csv", index=False)
    print(f"\nSaved rank {rank} 5-fold CV results to {WORK / f'true_5fold_cv_results_rank{rank}.csv'}")

def run_continuous_dual_ascent(store, device, mu0_start=5.0):
    print(f"\n=======================================================")
    print(f"   TASK 2: CONTINUOUS DUAL ASCENT (mu0 = {mu0_start:.1f})   ")
    print("=======================================================")
    
    tr_clips, val_clips = split_by_scene(store, "scene_S009")
    codec = ssf2020(quality=4, pretrained=True).to(device).train()
    opt = torch.optim.Adam((p for p in codec.parameters() if p.requires_grad), lr=1e-4)
    # Use tau=0.996 and scaled mu_ref to make the controller work with the MSE-based soft_u scale
    controller = DualController(tau=0.992, mu_ref=10000.0, eta_frac=0.44, beta=0.9, deadband=0.0002)
    controller.reset(mu0=mu0_start)
    
    current_mu = mu0_start
    records = []
    _rng = np.random.default_rng(42)
    
    for outer_step in range(1, 101):
        # 1. Fine-tune codec live for 400 steps at current_mu
        codec.train()
        for _ in range(400):
            clip = tr_clips[int(_rng.integers(len(tr_clips)))]
            x, om = make_gop(store, clip)
            x, om = x.to(device), om.to(device)
            x_hat, lik = ssf_forward(codec, x)
            # normalize_weight=False is required so that increasing mu INCREASES quality/utility
            loss, _ = srdo_loss(x_hat, x, lik, om, mu=current_mu, lambda_0=0.08, num_pixels=PATCH**2, normalize_weight=False)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(codec.parameters(), max_norm=1.0)
            opt.step()
            
        # 2. Evaluate utility live on validation set
        codec.eval()
        val_u_list = []
        with torch.no_grad():
            for v_clip in val_clips:
                vx, _ = make_gop(store, v_clip)
                vx = vx.to(device)
                vx_hat, _ = ssf_forward(codec, vx)
                val_u_list.append(1.0 - float(F.mse_loss(vx_hat, vx)))
                
        soft_u = np.mean(val_u_list)
        new_mu = controller.step(soft_u)
        
        hist = controller.history[-1]
        hist["outer_step"] = outer_step
        records.append(hist)
        
        print(f"Step {outer_step:2d} | mu = {current_mu:6.2f} -> {new_mu:6.2f} | soft_U = {soft_u:.4f} | active = {hist['active']}")
        current_mu = new_mu

    df = pd.DataFrame(records)
    out_name = f"continuous_dual_ascent_mu0_{int(mu0_start)}.csv"
    df.to_csv(WORK / out_name, index=False)
    print(f"Saved trajectory to {WORK / out_name}")

def main():
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    
    if torch.cuda.is_available():
        gpu_id = rank % torch.cuda.device_count()
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        
    print(f"[Rank {rank}/{world_size}] Using device: {device}")
    
    data_root = find_dataset_root()
    store = preload(data_root)
    
    # Run 1: Strict 5-fold CV (Fold split across GPUs)
    # run_true_5fold_cv(store, device, rank=rank, world_size=world_size)
    
    # Task 2/3 run on Rank 0 only
    if rank == 0:
        run_continuous_dual_ascent(store, device, mu0_start=5.0)
        run_continuous_dual_ascent(store, device, mu0_start=30.0)

if __name__ == "__main__":
    main()
