"""Standard-codec RD anchor on domain D, in rate, PSNR, and task utility.

This produces the reference curve the SRDO operating points are measured
against. Three decisions in it are not obvious and are worth stating, because
each one is a place a BD-BR number can be quietly wrong.

WHAT IS ENCODED. The anchor encodes the same 256x256 native-resolution crops
the neural codec consumes, not the full 1080p frame. Comparing a neural codec
evaluated on crops against a standard codec run on whole frames would hand the
standard codec a large spatial-context advantage and then attribute the
difference to the method. The cost of this choice is that a block codec at
256x256 has less context than it would in deployment, so the anchor is
pessimistic in absolute terms; it is the like-for-like comparison, and the
caveat belongs in the paper rather than in a silently different pipeline.

WHAT RATE MEANS. Rate is measured from the encoded bitstream on disk over the
exact frames evaluated, not from an encoder-reported average or a target. The
source files are MPEG-2 at ~0.52 bpp, high enough that this measurement is not
dominated by the source encoder — which is the property that makes domain D
usable for BD-BR at all and the existing corpus unusable.

WHAT QUALITY MEANS. Both PSNR and task utility are recorded at every operating
point. A codec tuned for a detector is expected to lose PSNR by construction,
so reporting only BD-BR(PSNR) would misstate the result in one direction and
reporting only BD-BR(U) would flatter it in the other. Utility uses the same
pseudo-ground-truth protocol as the rest of the paper: the frozen detector's
output on the uncompressed crop is the reference.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from srdo.bdrate import psnr  # noqa: E402
from srdo.detection import detect, evaluate as evaluate_detections  # noqa: E402
from srdo.video import probe  # noqa: E402

DATA = ROOT / "dataset-2"
OUT_CSV = ROOT / "outputs" / "phase5_anchor_rd.csv"

PATCH = 256
N_FRAMES = 64          # encoded and PSNR-scored
UTILITY_STRIDE = 4     # detector runs on every 4th frame: 16 per clip per point
CRFS = (20, 24, 28, 32, 36, 40)


def decode_window(path: pathlib.Path, start: int, n: int,
                  box: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Decode n frames, optionally cropped to (w, h, x, y), as (T,H,W,3) uint8."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path)]
    if box is not None:
        w, h, x, y = box
        cmd += ["-vf", f"crop={w}:{h}:{x}:{y}"]
    cmd += ["-frames:v", str(start + n),
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.decode()[-300:])

    meta = probe(path)
    w, h = (box[0], box[1]) if box is not None else (meta.width, meta.height)
    buf = np.frombuffer(res.stdout, dtype=np.uint8)
    fb = h * w * 3
    frames = buf[: (buf.size // fb) * fb].reshape(-1, h, w, 3)
    return frames[start:start + n]


def choose_crop(path: pathlib.Path, start: int, n: int,
                probe_stride: int = 8) -> tuple[tuple[int, int, int, int], float]:
    """Pick the PATCH-sized window to evaluate, from the clean video only.

    A 256x256 window is 1.7% of a 1080p frame, so a fixed centre crop lands on
    empty floor for most surveillance framings and yields sequences with no
    subject to preserve — on which every codec scores identically and the
    comparison measures nothing. The window is therefore placed at the centroid
    of the detections in the uncompressed sequence.

    The selection depends only on the source, never on any codec or operating
    point, and the same window is then used for every CRF and for the neural
    codec, so it cannot favour either. Returns the box and the fraction of
    probed frames containing a detection inside it.
    """
    meta = probe(path)
    frames = decode_window(path, start, n)[::probe_stride]
    det = detect(frames)

    centres = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
               for boxes in det.boxes for b in boxes]
    if centres:
        cx, cy = (float(np.median([c[0] for c in centres])),
                  float(np.median([c[1] for c in centres])))
    else:
        cx, cy = meta.width / 2.0, meta.height / 2.0

    x = int(np.clip(cx - PATCH / 2, 0, meta.width - PATCH))
    y = int(np.clip(cy - PATCH / 2, 0, meta.height - PATCH))
    x -= x % 2
    y -= y % 2

    inside = [
        any(min(b[2], x + PATCH) > max(b[0], x) and min(b[3], y + PATCH) > max(b[1], y)
            for b in boxes)
        for boxes in det.boxes
    ]
    return (PATCH, PATCH, x, y), float(np.mean(inside)) if inside else 0.0


def encode_decode_measure(frames: np.ndarray, crf: int, codec: str,
                          fps: float) -> tuple[np.ndarray, float]:
    """Round-trip through a real encoder. Returns (decoded frames, bits-per-pixel).

    Rate comes from the size of the bitstream file, which is what a decoder
    would actually have to receive.
    """
    t, h, w, _ = frames.shape
    with tempfile.TemporaryDirectory() as td:
        bitstream = pathlib.Path(td) / "out.mp4"

        enc = subprocess.run(
            ["ffmpeg", "-v", "error", "-y",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
             "-r", f"{fps}", "-i", "-",
             "-c:v", codec, "-crf", str(crf), "-preset", "medium",
             "-pix_fmt", "yuv420p", str(bitstream)],
            input=frames.tobytes(), capture_output=True)
        if enc.returncode != 0:
            raise RuntimeError(enc.stderr.decode()[-300:])

        bits = bitstream.stat().st_size * 8
        bpp = bits / (t * h * w)

        dec = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(bitstream),
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            capture_output=True)
        if dec.returncode != 0:
            raise RuntimeError(dec.stderr.decode()[-300:])

    buf = np.frombuffer(dec.stdout, dtype=np.uint8)
    fb = h * w * 3
    out = buf[: (buf.size // fb) * fb].reshape(-1, h, w, 3)
    if out.shape[0] < t:
        raise RuntimeError(f"decoded {out.shape[0]} of {t} frames")
    return out[:t], bpp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec", default="libx265",
                    help="ffmpeg encoder; libx265 (HEVC) until vvenc is available")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    clips = sorted(DATA.rglob("*.mpeg"))
    if args.limit:
        clips = clips[: args.limit]
    print(f"{len(clips)} clips x {len(CRFS)} operating points, "
          f"{N_FRAMES} frames each, codec={args.codec}\n")

    rows, t0 = [], time.time()
    for ci, path in enumerate(clips):
        meta = probe(path)
        # ffprobe's nb_frames overstates length on some MPEG-2 files in this
        # corpus, so a midpoint start can land past the last decodable frame.
        # Fall back to the head rather than dropping the clip.
        clean, box, occupancy = None, None, 0.0
        for start in (max((meta.nb_frames - N_FRAMES) // 2, 0), 0):
            try:
                box, occupancy = choose_crop(path, start, N_FRAMES)
                candidate = decode_window(path, start, N_FRAMES, box)
            except Exception as exc:
                print(f"  {path.name} @{start}: {type(exc).__name__}: {exc}")
                continue
            if candidate.shape[0] >= N_FRAMES:
                clean = candidate
                break
            print(f"  {path.name} @{start}: only {candidate.shape[0]} frames")

        if clean is None:
            print(f"  skip {path.name}: no decodable {N_FRAMES}-frame window")
            continue

        # Pseudo-GT once per clip, on the uncompressed crop.
        sub = clean[::UTILITY_STRIDE]
        ref = detect(sub)
        if ref.n_total == 0:
            print(f"  skip {path.name}: no detections in the selected window")
            continue
        print(f"  {path.name}: window at ({box[2]},{box[3]}), "
              f"subject present in {occupancy:.0%} of probed frames")

        for crf in CRFS:
            try:
                dec, bpp = encode_decode_measure(clean, crf, args.codec, meta.fps)
            except Exception as exc:
                print(f"  {path.name} crf{crf}: {type(exc).__name__}: {exc}")
                continue

            mse = float(np.mean((dec.astype(np.float32) / 255.0
                                 - clean.astype(np.float32) / 255.0) ** 2))
            hard_u, soft_u = evaluate_detections(ref, detect(dec[::UTILITY_STRIDE]))

            rows.append({
                "clip": path.name, "scene": path.parent.name,
                "codec": args.codec, "crf": crf,
                "bpp": bpp, "mse": mse, "psnr_db": float(psnr(mse)),
                "soft_u": float(soft_u), "hard_u": float(hard_u),
                "n_gt": int(ref.n_total), "n_frames": int(clean.shape[0]),
                "src_bpp": round(meta.bpp, 4),
                "crop_x": box[2], "crop_y": box[3], "occupancy": round(occupancy, 3),
            })
            r = rows[-1]
            print(f"  {path.name:26s} crf{crf:3d}  bpp={r['bpp']:.4f}  "
                  f"psnr={r['psnr_db']:5.2f}  U={r['soft_u']:.3f}")

        print(f"[{ci + 1}/{len(clips)}] {time.time() - t0:.0f}s elapsed")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}  ({len(df)} operating points)\n")

    if df.empty:
        return

    agg = df.groupby("crf").agg(
        bpp=("bpp", "mean"), psnr_db=("psnr_db", "mean"),
        soft_u=("soft_u", "mean"), hard_u=("hard_u", "mean"), n=("clip", "size"))
    print(agg.round(4).to_string())
    print("\nThis is the anchor curve. BD-BR requires the SRDO points on the same "
          "content; compute both with srdo.bdrate once those exist.")


if __name__ == "__main__":
    main()
