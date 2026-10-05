"""Standard-codec anchor on EXACTLY the GOPs the neural codec was validated on.

The first anchor (`run_phase5_anchor.py`) placed one window per clip and encoded
64-frame sequences. The domain D neural sweep places a window per GOP and
evaluates 8-frame GOPs from a held-out scene. Comparing those two directly
gives the neural codec an easier validation set — per-GOP windows keep the
subject centred, which is exactly the condition a detector does best on — and
the resulting 0.12 utility gap is uninterpretable.

This reproduces the neural run's validation set bit for bit: same scene, same
per-GOP window selection from the uncompressed video, same 8-frame GOPs, same
evenly spaced indices. The window logic is duplicated here rather than imported
because the Kaggle payload is not on the local path; it is deterministic given
the source, so both pipelines land on identical pixels.

Only then is BD-BR meaningful. Two caveats survive and belong in the text:
8-frame sequences amortise a keyframe poorly, which penalises both codecs but
not necessarily equally; and 256x256 inputs give a block codec far less spatial
context than deployment would.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from srdo.bdrate import RDCurve, bd_rate, overlap_fraction, psnr  # noqa: E402
from srdo.detection import detect, evaluate as evaluate_detections  # noqa: E402

DATA = ROOT / "dataset-2"
OUT_CSV = ROOT / "outputs" / "phase5_anchor_matched.csv"

PATCH, GOP = 256, 8
VAL_SCENE = "scene_S009"
VAL_WINDOWS_PER_CLIP = 8
CRFS = (20, 24, 28, 32, 36, 40)


def _probe_size(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
        capture_output=True, text=True).stdout.strip()
    a = out.split("x")
    return int(a[0]), int(a[1])


def _decode(path, box=None, max_frames=None):
    w0, h0 = _probe_size(path)
    cmd = ["ffmpeg", "-v", "error", "-i", str(path)]
    if box is not None:
        w, h, x, y = box
        cmd += ["-vf", f"crop={w}:{h}:{x}:{y}"]
    else:
        w, h = w0, h0
    if max_frames:
        cmd += ["-frames:v", str(max_frames)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.decode()[-300:])
    buf = np.frombuffer(res.stdout, dtype=np.uint8)
    fb = h * w * 3
    return buf[: (buf.size // fb) * fb].reshape(-1, h, w, 3)


def gop_windows(full, stride_probe=GOP):
    """Identical to the neural runner's selection, so the GOPs match exactly."""
    t, h0, w0 = full.shape[:3]
    probe_idx = list(range(stride_probe // 2, t, stride_probe))
    det = detect(full[probe_idx])
    out = {}
    for j, i in enumerate(probe_idx):
        boxes = det.boxes[j]
        if len(boxes) == 0:
            continue
        cx = float(np.median([(b[0] + b[2]) / 2 for b in boxes]))
        cy = float(np.median([(b[1] + b[3]) / 2 for b in boxes]))
        x = int(np.clip(cx - PATCH / 2, 0, w0 - PATCH)) & ~1
        y = int(np.clip(cy - PATCH / 2, 0, h0 - PATCH)) & ~1
        out[max(min(i - stride_probe // 2, t - GOP), 0)] = (x, y)
    return out


def build_val_gops():
    """The neural run's validation set: held-out scene, evenly spaced GOPs."""
    clips = sorted(p for p in DATA.rglob("*.mpeg")
                   if p.name.rsplit("_cam", 1)[0] == VAL_SCENE)
    val = []
    for p in clips:
        full = _decode(p)
        gops = [full[s:s + GOP, y:y + PATCH, x:x + PATCH].copy()
                for s, (x, y) in sorted(gop_windows(full).items())
                if full[s:s + GOP, y:y + PATCH, x:x + PATCH].shape[:3] == (GOP, PATCH, PATCH)]
        del full
        k = min(VAL_WINDOWS_PER_CLIP, len(gops))
        for i in np.linspace(0, len(gops) - 1, k).round().astype(int):
            val.append((f"{p.name}#{i}", gops[int(i)]))
        print(f"  {p.name}: {len(gops)} usable GOPs -> {k} sampled")
    return val


def encode_decode(frames, crf, codec, fps=29.97):
    t, h, w, _ = frames.shape
    with tempfile.TemporaryDirectory() as td:
        bs = pathlib.Path(td) / "out.mp4"
        enc = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", codec,
             "-crf", str(crf), "-preset", "medium", "-pix_fmt", "yuv420p", str(bs)],
            input=frames.tobytes(), capture_output=True)
        if enc.returncode != 0:
            raise RuntimeError(enc.stderr.decode()[-300:])
        bpp = bs.stat().st_size * 8 / (t * h * w)
        dec = subprocess.run(["ffmpeg", "-v", "error", "-i", str(bs), "-f", "rawvideo",
                              "-pix_fmt", "rgb24", "-"], capture_output=True)
    buf = np.frombuffer(dec.stdout, dtype=np.uint8)
    fb = h * w * 3
    return buf[: (buf.size // fb) * fb].reshape(-1, h, w, 3)[:t], bpp


def main():
    codec = sys.argv[1] if len(sys.argv) > 1 else "libx265"
    print(f"building the neural run's validation GOPs from {VAL_SCENE}")
    val = build_val_gops()
    print(f"{len(val)} GOPs, codec={codec}\n")

    rows = []
    for name, clean in val:
        ref = detect(clean)
        if ref.n_total == 0:
            continue                       # same exclusion the neural eval applies
        for crf in CRFS:
            dec, bpp = encode_decode(clean, crf, codec)
            mse = float(np.mean((dec.astype(np.float32) / 255
                                 - clean.astype(np.float32) / 255) ** 2))
            hard_u, soft_u = evaluate_detections(ref, detect(dec))
            rows.append({"gop": name, "crf": crf, "bpp": bpp, "mse": mse,
                         "psnr_db": float(psnr(mse)), "soft_u": float(soft_u),
                         "hard_u": float(hard_u), "n_gt": int(ref.n_total)})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    g = df.groupby("crf").agg(bpp=("bpp", "mean"), psnr=("psnr_db", "mean"),
                              U=("soft_u", "mean"), hard=("hard_u", "mean"),
                              n=("gop", "size")).sort_values("bpp")
    print(f"\nanchor on the matched GOPs ({df['gop'].nunique()} scored)\n{g.round(4).to_string()}")

    ssf = pd.read_csv(ROOT / "notebooks/phase4-cctv/v8/phase4d_fixed_mu_sweep_d.csv")
    print(f"\nSSF on the same GOPs:\n{ssf[['mu','bpp','soft_u']].round(4).to_string(index=False)}")
    print("\nSSF vs anchor at matched rate:")
    for _, r in ssf.iterrows():
        ua = float(np.interp(r.bpp, g.bpp, g.U))
        print(f"  mu={r.mu:4.0f}: SSF {r.soft_u:.3f} @ {r.bpp:.4f} | anchor {ua:.3f} | {r.soft_u - ua:+.3f}")

    # BD-BR needs >=4 points and a monotone quality axis on each curve.
    try:
        anc_u = RDCurve("x265", g.bpp.values, g.U.values)
        ssf_u = RDCurve("SSF", ssf.bpp.values, ssf.soft_u.values)
        print(f"\nBD-Rate (utility axis): {bd_rate(anc_u, ssf_u):+.1f}%  "
              f"overlap {overlap_fraction(anc_u, ssf_u):.2f}")
    except Exception as exc:
        print(f"\nBD-Rate(U) not computable: {exc}")
        print("  the SSF points span a narrow utility range by construction, so a "
              "BD number over that interval would describe almost nothing")


if __name__ == "__main__":
    main()
