#!/usr/bin/env python3
"""Visualize a single CoastalBench ocean_met field from a local Zarr day file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import zarr

FIELD_METADATA = {
    # 3D vars: 12 vertical levels each, stacked as separate channels
    "u":     {"base": 0,  "n_levels": 12, "cmap": "RdBu_r",  "signed": True},
    "v":     {"base": 12, "n_levels": 12, "cmap": "RdBu_r",  "signed": True},
    "w":     {"base": 24, "n_levels": 12, "cmap": "RdBu_r",  "signed": True},
    "temp":  {"base": 36, "n_levels": 12, "cmap": "viridis", "signed": False},
    "salt":  {"base": 48, "n_levels": 12, "cmap": "viridis", "signed": False},
    "AKs":   {"base": 60, "n_levels": 12, "cmap": "magma",   "signed": False},
    # 2D surface / met + mask
    "zeta":  {"base": 72, "n_levels": 1,  "cmap": "RdBu_r",  "signed": True},
    "Pair":  {"base": 73, "n_levels": 1,  "cmap": "viridis", "signed": False},
    "Qair":  {"base": 74, "n_levels": 1,  "cmap": "viridis", "signed": False},
    "Tair":  {"base": 75, "n_levels": 1,  "cmap": "viridis", "signed": False},
    "rain":  {"base": 76, "n_levels": 1,  "cmap": "Blues",   "signed": False},
    "swrad": {"base": 77, "n_levels": 1,  "cmap": "inferno", "signed": False},
    "lwrad": {"base": 78, "n_levels": 1,  "cmap": "inferno", "signed": False},
    "Uwind": {"base": 79, "n_levels": 1,  "cmap": "RdBu_r",  "signed": True},
    "Vwind": {"base": 80, "n_levels": 1,  "cmap": "RdBu_r",  "signed": True},
    "mask":  {"base": 81, "n_levels": 1,  "cmap": "gray",    "signed": False},
}

N_FEATURES = 82
N_TIMES = 48


def build_idx_to_meta(field_meta: dict) -> dict[int, dict]:
    """idx -> {name, level, label, cmap, signed}."""
    out: dict[int, dict] = {}
    for name, meta in field_meta.items():
        for level in range(meta["n_levels"]):
            idx = meta["base"] + level
            label = name if meta["n_levels"] == 1 else f"{name}_z{level}"
            out[idx] = {
                "name": name,
                "level": level,
                "label": label,
                "cmap": meta["cmap"],
                "signed": meta["signed"],
            }
    return out


IDX_TO_META = build_idx_to_meta(FIELD_METADATA)
assert set(IDX_TO_META) == set(range(N_FEATURES)), "FIELD_METADATA does not cover indices 0..81"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    # this script is in the scripts directory, so we need to go up two levels to get to the root
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root / "data" / "CoastalBench-preview",
        help="Path to the CoastalBench-preview root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "figures" / "fields",
        help="Directory for saved figures",
    )
    parser.add_argument(
        "--field-idx",
        type=int,
        default=None,
        help="Feature index in [0, 81]",
    )
    parser.add_argument(
        "--field-name",
        type=str,
        default=None,
        help=f"Field name: {', '.join(FIELD_METADATA)}",
    )
    parser.add_argument(
        "--field-level",
        type=int,
        default=None,
        help="Vertical level for 3D fields (default: 0)",
    )
    parser.add_argument(
        "--day",
        type=str,
        default="20080102",
        help="Day folder name YYYYMMDD (default: %(default)s)",
    )
    parser.add_argument(
        "--time-idx",
        type=int,
        default=0,
        help="Half-hour index in [0, 47] (default: %(default)s)",
    )
    return parser.parse_args()


def resolve_field(name: str, level: int = 0) -> tuple[int, str, str, bool]:
    if name not in FIELD_METADATA:
        known = ", ".join(FIELD_METADATA)
        raise ValueError(f"Unknown field-name {name!r}. Known: {known}")

    meta = FIELD_METADATA[name]
    if not (0 <= level < meta["n_levels"]):
        raise ValueError(
            f"Level {level} out of range for {name} "
            f"(expected 0..{meta['n_levels'] - 1})"
        )

    idx = meta["base"] + level
    label = name if meta["n_levels"] == 1 else f"{name}_z{level}"

    return idx, label, meta["cmap"], meta["signed"]


def resolve_from_args(args: argparse.Namespace) -> tuple[int, str, str, bool]:
    if args.field_idx is not None:
        if args.field_idx not in IDX_TO_META:
            raise ValueError(f"--field-idx must be in 0..{N_FEATURES - 1}")
        meta = IDX_TO_META[args.field_idx]
        return args.field_idx, meta["label"], meta["cmap"], meta["signed"]
    if args.field_name is not None:
        level = 0 if args.field_level is None else args.field_level
        return resolve_field(args.field_name, level)
    raise ValueError("Provide --field-idx or --field-name.")


def load_field(
    data_dir: Path,
    day: str,
    time_idx: int,
    field_idx: int,
    apply_ocean_mask: bool,
) -> np.ma.MaskedArray | np.ndarray:
    if not (0 <= time_idx < N_TIMES):
        raise ValueError(f"--time-idx must be in 0..{N_TIMES - 1}, got {time_idx}")

    zarr_path = data_dir / "ocean_met" / f"{day}.zarr"
    if not zarr_path.is_dir():
        raise FileNotFoundError(f"Missing day Zarr: {zarr_path}")

    arr = zarr.open(str(zarr_path), mode="r")
    field = np.asarray(arr[field_idx, :, :, time_idx], dtype=np.float32)

    if not apply_ocean_mask:
        return field

    landsea = np.asarray(arr[81, :, :, time_idx], dtype=np.float32)
    ocean = np.abs(landsea - 1.0) <= 1e-3
    return np.ma.array(field, mask=~ocean)


def color_limits(
    field: np.ndarray | np.ma.MaskedArray,
    signed: bool,
    pct: float = 99.0,
) -> tuple[float | None, float | None]:
    data = np.asarray(
        field.compressed() if np.ma.isMaskedArray(field) else field
    )

    data = data[np.isfinite(data)]
    if data.size == 0:
        return None, None
    if signed:
        lim = float(np.percentile(np.abs(data), pct))
        lim = lim if lim > 0 else 1.0
        return -lim, lim
    lo, hi = np.percentile(data, [100.0 - pct, pct])
    if lo == hi:
        hi = float(lo) + 1.0
    return float(lo), float(hi)


def save_field(
    field: np.ndarray | np.ma.MaskedArray,
    output_dir: Path,
    day: str,
    time_idx: int,
    label: str,
    cmap: str,
    signed: bool,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    vmin, vmax = color_limits(field, signed=signed)

    title = f"{label}, day {day}, t={time_idx}"
    written: list[Path] = []

    for suffix in ("png", "pdf"):
        path = output_dir / f"{label}_{day}_t{time_idx}.{suffix}"
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(
            field,
            origin="lower",
            cmap=cmap,
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("width")
        ax.set_ylabel("height")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)

    return written


def main() -> int:
    args = parse_args()
    try:
        field_idx, label, cmap, signed = resolve_from_args(args)
        apply_mask = IDX_TO_META[field_idx]["name"] != "mask"
        field = load_field(
            args.data_dir,
            args.day,
            args.time_idx,
            field_idx,
            apply_ocean_mask=apply_mask,
        )
        written = save_field(
            field,
            args.output_dir,
            args.day,
            args.time_idx,
            label,
            cmap,
            signed,
        )
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved {label} (idx={field_idx}, signed={signed})")
    for path in written:
        print(f"  {path}")
    return 0
    

if __name__ == "__main__":
    raise SystemExit(main())
