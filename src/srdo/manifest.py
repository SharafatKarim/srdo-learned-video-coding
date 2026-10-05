"""Phase 0 — dataset characterisation and usable-subset selection.

Produces the dataset table for the paper and the clean subset every later phase
runs on. Two things are decided here and nowhere else:

  1. the crop rectangle that removes pillarboxing (see video.detect_crop), and
  2. whether a clip is usable at all, which is governed by the latent-grid
     argument in Phase0Config.min_crop_height.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .config import CONFIG, MANIFEST_CSV, Phase0Config, ensure_dirs
from .video import Crop, detect_crop, probe


def _source_of(name: str) -> str:
    """Clip names are '<SOURCE>_<clip-id>_<class>_<n>.mp4'."""
    return name.split("_", 1)[0]


def _scan_one(path_str: str, cfg: Phase0Config) -> dict | None:
    path = Path(path_str)
    try:
        meta = probe(path)
    except Exception as exc:  # noqa: BLE001 - one bad file must not kill the scan
        return {"path": path_str, "error": str(exc)}

    crop = detect_crop(path, cfg.crop_limit, cfg.crop_round, cfg.crop_probe_frames)
    if crop is None:
        crop = Crop(meta.width, meta.height, 0, 0)

    # Fraction of the frame that cropdetect judged to be border. Large values
    # flag clips whose apparent bitrate is mostly spent on constant black.
    src_area = max(meta.width * meta.height, 1)
    border_frac = 1.0 - (crop.w * crop.h) / src_area

    long_enough = meta.nb_frames >= cfg.min_frames and meta.fps > 0
    usable_codec = long_enough and crop.h >= cfg.min_crop_height
    usable_probe = long_enough and crop.h >= cfg.min_probe_height

    return {
        "path": path_str,
        "clip": path.name,
        "action": path.parent.name,
        "source": _source_of(path.name),
        "codec": meta.codec,
        "width": meta.width,
        "height": meta.height,
        "fps": meta.fps,
        "nb_frames": meta.nb_frames,
        "duration": meta.duration,
        "bit_rate": meta.bit_rate,
        "src_bpp": meta.bpp,
        "crop_w": crop.w,
        "crop_h": crop.h,
        "crop_x": crop.x,
        "crop_y": crop.y,
        "border_frac": border_frac,
        "usable_codec": usable_codec,
        "usable_probe": usable_probe,
        # `usable` is the default tier for phases 1-2.
        "usable": usable_probe,
        "error": "",
    }


def build_manifest(
    data_root: Path | None = None,
    cfg: Phase0Config | None = None,
    workers: int = 8,
) -> pd.DataFrame:
    cfg = cfg or CONFIG.phase0
    data_root = data_root or cfg.data_root

    paths = sorted(str(p) for p in data_root.rglob("*.mp4"))
    if not paths:
        raise FileNotFoundError(f"no .mp4 files under {data_root}")

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_scan_one, p, cfg): p for p in paths}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="phase0 scan"):
            row = fut.result()
            if row is not None:
                rows.append(row)

    df = pd.DataFrame(rows).sort_values(["action", "clip"]).reset_index(drop=True)
    ensure_dirs()
    df.to_csv(MANIFEST_CSV, index=False)
    return df


def load_manifest() -> pd.DataFrame:
    if not MANIFEST_CSV.exists():
        raise FileNotFoundError(f"{MANIFEST_CSV} missing — run phase 0 first")
    return pd.read_csv(MANIFEST_CSV)


def summarise(df: pd.DataFrame) -> str:
    """Human-readable summary; the numbers here go straight into the paper."""
    ok = df[df["error"] == ""] if "error" in df else df
    probe_t = ok[ok["usable_probe"]]
    codec_t = ok[ok["usable_codec"]]
    n = max(len(ok), 1)

    lines = [
        "=== Phase 0: dataset characterisation ===",
        f"clips scanned        : {len(df)}",
        f"probe failures       : {(df['error'] != '').sum()}",
        f"usable (probe tier)  : {len(probe_t)}  ({100 * len(probe_t) / n:.1f}%)   phases 1-2, CPU",
        f"usable (codec tier)  : {len(codec_t)}  ({100 * len(codec_t) / n:.1f}%)   phases 4-5, GPU",
        "",
        "--- by source ---",
        ok.groupby("source").agg(
            clips=("clip", "size"),
            probe_tier=("usable_probe", "sum"),
            codec_tier=("usable_codec", "sum"),
            median_bpp=("src_bpp", "median"),
            median_h=("crop_h", "median"),
        ).to_string(),
        "",
        "--- class coverage ---",
        pd.DataFrame({
            "clips": ok.groupby("action").size(),
            "probe_tier": ok.groupby("action")["usable_probe"].sum(),
            "codec_tier": ok.groupby("action")["usable_codec"].sum(),
        }).to_string(),
        "",
        "--- resolution after crop (probe tier) ---",
        probe_t.groupby(["crop_w", "crop_h"]).size()
        .sort_values(ascending=False).head(8).to_string(),
        "",
        f"median source bpp (all)        : {ok['src_bpp'].median():.4f}",
        f"median source bpp (codec tier) : {codec_t['src_bpp'].median():.4f}",
        f"clips with >5% border          : {(ok['border_frac'] > 0.05).sum()}",
        f"duration, probe tier           : {probe_t['duration'].sum() / 60:.1f} min",
        f"duration, codec tier           : {codec_t['duration'].sum() / 60:.1f} min",
        "",
        f"classes with zero probe-tier clips : "
        f"{sorted(set(ok['action']) - set(probe_t['action'])) or 'none'}",
        f"classes with zero codec-tier clips : "
        f"{sorted(set(ok['action']) - set(codec_t['action'])) or 'none'}",
    ]
    return "\n".join(lines)
