#!/usr/bin/env python
"""
Cross-spectral coherence evaluation for CoastalBench (data characterization).

This script evaluates the CoastalBench dataset for cross-spectral coherence
properties WITHOUT a trained model. It produces GT-only panels that characterize
whether the dataset is suitable for training with the physical coherence loss term.

Panels produced
---------------
  A  Cross-field coherence matrix (GT-only, per scale band)
  B  Coherence curves γ²(ν) for selected field pairs
  C  Per-field auto-spectrum (log scale)
  D  Cross-band coupling matrices Q̂ for top pairs
  E  Ensemble-averaged spatial Pearson correlation heatmap
  F  Distribution of pairwise band-averaged γ² (histogram)
  G  Temporal coherence: consecutive-day self-coherence per field
  H  Band energy coefficient of variation across the ensemble
  I  Auto-spectrum comparison: full ocean vs. coastal-buffer-eroded mask
     (only produced when --coastal-buffer > 0)

Usage
-----
  # Evaluate default surface+thermodynamics fields over 30 days at midday (t=12):
  python xspec_coherence.py \\
      --data /path/to/CoastalBench-preview \\
      --fields u_z0 v_z0 zeta temp_z0 Tair swrad Uwind Vwind \\
      --ensemble-size 30 --time 12

  # Vertical structure subset with coastal erosion:
  python xspec_coherence.py \\
      --fields u_z0 u_z6 u_z11 v_z0 v_z6 v_z11 \\
      --coastal-buffer 10

  # List all valid field labels:
  python xspec_coherence.py --list-fields

Field label convention
----------------------
  3D fields use name_zL notation: u_z0, u_z6, temp_z0, AKs_z11, ...
  2D fields use plain name:        zeta, Tair, Uwind, swrad, ...
  (Run --list-fields for the full set.)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import zarr

# ============================================================
# TC graph-spectral computation — imported via sys.path
# ============================================================
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TC_SRC = _REPO_ROOT / "0_demo_TurbulentCombustion" / "src"
if str(_TC_SRC) not in sys.path:
    sys.path.insert(0, str(_TC_SRC))

from cross_spectral_coherence import (       # noqa: E402
    compute_graph_basis,
    GraphSpectralBasis,
    accumulate_spectra,
    SpectralEstimate,
    coherence_from_bins,
    band_coherence,
    coherence_matrix,
    frequency_bins,
    compute_band_energies,
    cross_band_coupling_scores,
    BandEnergyEstimate,
    project_fields,
    BAND_TINT,
    BAND_EDGE_C,
    EPS,
)
import paper_style as ps                     # noqa: E402  (TC src dir on path)

# ============================================================
# Field metadata  (matches field_visualization.py exactly)
# ============================================================
FIELD_METADATA: Dict[str, dict] = {
    # 3D vars: 12 vertical levels each, stored as separate 2-D channels
    "u":     {"base": 0,  "n_levels": 12, "cmap": "RdBu_r",  "signed": True},
    "v":     {"base": 12, "n_levels": 12, "cmap": "RdBu_r",  "signed": True},
    "w":     {"base": 24, "n_levels": 12, "cmap": "RdBu_r",  "signed": True},
    "temp":  {"base": 36, "n_levels": 12, "cmap": "viridis", "signed": False},
    "salt":  {"base": 48, "n_levels": 12, "cmap": "viridis", "signed": False},
    "AKs":   {"base": 60, "n_levels": 12, "cmap": "magma",   "signed": False},
    # 2D surface / meteorological fields
    "zeta":  {"base": 72, "n_levels": 1,  "cmap": "RdBu_r",  "signed": True},
    "Pair":  {"base": 73, "n_levels": 1,  "cmap": "viridis", "signed": False},
    "Qair":  {"base": 74, "n_levels": 1,  "cmap": "viridis", "signed": False},
    "Tair":  {"base": 75, "n_levels": 1,  "cmap": "viridis", "signed": False},
    "rain":  {"base": 76, "n_levels": 1,  "cmap": "Blues",   "signed": False},
    "swrad": {"base": 77, "n_levels": 1,  "cmap": "inferno", "signed": False},
    "lwrad": {"base": 78, "n_levels": 1,  "cmap": "inferno", "signed": False},
    "Uwind": {"base": 79, "n_levels": 1,  "cmap": "RdBu_r",  "signed": True},
    "Vwind": {"base": 80, "n_levels": 1,  "cmap": "RdBu_r",  "signed": True},
    # index 81 is the land-sea mask; excluded from analysis fields
}

MASK_IDX = 81
N_TIMES   = 48   # half-hourly steps per day


def build_idx_to_meta(field_meta: Dict[str, dict]) -> Dict[int, dict]:
    """Feature index -> {name, level, label, cmap, signed}."""
    out: Dict[int, dict] = {}
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


IDX_TO_META: Dict[int, dict] = build_idx_to_meta(FIELD_METADATA)

# label -> feature index lookup, built at module load time
_LABEL_TO_IDX: Dict[str, int] = {
    meta["label"]: idx for idx, meta in IDX_TO_META.items()
}


def parse_fields(tokens: List[str]) -> Tuple[List[int], List[str]]:
    """Resolve field labels (e.g. 'u_z6', 'zeta') to feature indices and labels.

    Raises SystemExit with a helpful message on unknown tokens.
    """
    indices, labels = [], []
    for tok in tokens:
        if tok not in _LABEL_TO_IDX:
            known = sorted(_LABEL_TO_IDX)
            raise SystemExit(
                f"Unknown field label {tok!r}. Known labels:\n  "
                + "\n  ".join(known)
            )
        indices.append(_LABEL_TO_IDX[tok])
        labels.append(tok)
    return indices, labels


# ============================================================
# CLI arguments
# ============================================================
def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(
        description=(
            "Cross-spectral coherence evaluation for the CoastalBench dataset. "
            "Produces GT-only characterization panels (no model required)."
        )
    )

    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="(Future) Trained model run dir or .pt file. Not used in data-only mode.",
    )
    p.add_argument(
        "--data",
        type=Path,
        default=repo_root / "data" / "CoastalBench-preview",
        help="Path to the CoastalBench-preview root directory.",
    )
    # 3D fields: name_zL notation (e.g. u_z0, u_z6, temp_z11).
    # 2D fields: plain name (e.g. zeta, Tair, swrad).
    # Run --list-fields for the full catalogue.
    p.add_argument(
        "--fields",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Field labels to evaluate. 3D fields: name_zL (e.g. u_z0, temp_z6). "
            "2D fields: plain name (e.g. zeta, Tair). "
            "Default: a surface-dynamics + thermodynamics subset."
        ),
    )
    p.add_argument(
        "--list-fields",
        action="store_true",
        help="Print all valid field labels and exit.",
    )
    p.add_argument(
        "--ensemble-size",
        type=int,
        default=24,
        help=(
            "Number of daily snapshots used for coherence estimation. "
            "The 90-day preview supports up to 90."
        ),
    )
    p.add_argument(
        "--day-stride",
        type=int,
        default=1,
        help=(
            "Sample every N-th available day to reduce temporal autocorrelation "
            "(default: 1 = use consecutive days)."
        ),
    )
    p.add_argument(
        "--time",
        type=int,
        default=12,
        help=(
            "Half-hourly time index in [0, 47] applied uniformly across all days "
            "(default: 12 = 6 h into the day)."
        ),
    )
    p.add_argument(
        "--n-modes",
        type=int,
        default=256,
        help="Number of graph-Laplacian eigenmodes (including the DC mode).",
    )
    p.add_argument(
        "--k-neighbors",
        type=int,
        default=12,
        help="Number of k-NN neighbours when building the graph Laplacian.",
    )
    p.add_argument(
        "--n-freq-bins",
        type=int,
        default=24,
        help="Number of frequency bins for coherence-spectrum curves.",
    )
    p.add_argument(
        "--basis-cache",
        type=Path,
        default=None,
        help="Path to cache/reuse the graph-Laplacian eigenbasis (.npz).",
    )
    p.add_argument(
        "--coastal-buffer",
        type=int,
        default=0,
        help=(
            "Erode the ocean mask by this many pixels from the coastline "
            "before building the graph. 0 = no erosion (default). "
            "When > 0, Panel I compares auto-spectra with and without the buffer."
        ),
    )
    p.add_argument(
        "--pairs",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Explicit field pairs to highlight in Panels B and D, given as "
            "comma-separated labels, e.g. --pairs u_z0,zeta temp_z0,Tair. "
            "Default: top-6 most coherent pairs by band-averaged γ²."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for figures (default: figures/xspec_eval/<timestamp>).",
    )

    return p.parse_args()


# ============================================================
# Data loading
# ============================================================
# --- NetCDF configuration -------------------------------------------------
#
# Mapping: FIELD_METADATA key → NetCDF variable name in the daily .nc file.
NC_FIELD_MAP: Dict[str, str] = {
    "u":     "u",
    "v":     "v",
    "w":     "w",
    "temp":  "temp",
    "salt":  "salt",
    "AKs":   "AKs",
    "zeta":  "zeta",
    "Pair":  "Pair",
    "Qair":  "Qair",
    "Tair":  "Tair",
    "rain":  "rain",
    "swrad": "swrad",
    "lwrad": "lwrad",
    "Uwind": "Uwind",
    "Vwind": "Vwind",
}

# Dimension names — update if necessary
NC_TIME_DIM  = "ocean_time"  # time dimension in daily files (length = 48)
NC_LEVEL_DIM = "s"           # vertical level dimension for 3-D fields (length = 12)

# Land-sea mask configuration.
# NC_MASK_FILE: 'static' → read from static.nc; 'daily' → read from each daily file.
# NC_MASK_VAR : variable name of the 2-D mask array (1 = ocean, 0 = land).
NC_MASK_FILE = "static"
NC_MASK_VAR  = "mask_rho"
# -------------------------------------------------------------------------

try:
    import xarray as xr  # required for NetCDF path only
    _XR_AVAILABLE = True
except ImportError:
    _XR_AVAILABLE = False


def _nc_engine() -> str:
    """Return the best available xarray NetCDF engine in this environment."""
    try:
        import xarray as xr
        available = list(xr.backends.list_engines().keys())
    except Exception:
        available = []
    for eng in ("netcdf4", "h5netcdf", "scipy"):
        if eng in available:
            return eng
    raise RuntimeError(
        f"No NetCDF engine available. xarray engines found: {available}.\n"
        "Install one: conda install -n PhyCoEnv netcdf4"
    )


def _detect_data_format(data_dir: Path) -> str:
    """Return 'zarr' or 'netcdf' based on the contents of *data_dir*.

    zarr  : data_dir/ocean_met/<date>.zarr/ structure (HuggingFace preview).
    netcdf: data_dir/<date>.nc flat layout    (full-precision OSN files).
    """
    if (data_dir / "ocean_met").is_dir():
        return "zarr"
    if any(data_dir.glob("[0-9]*.nc")):
        return "netcdf"
    raise FileNotFoundError(
        f"No recognised CoastalBench data found in {data_dir}.\n"
        f"Expected either data_dir/ocean_met/*.zarr (zarr preview) or "
        f"data_dir/YYYYMMDD.nc (full-precision NetCDF)."
    )


# ---------------------------------------------------------------------------
# Zarr loading (HuggingFace preview, float16)
# ---------------------------------------------------------------------------

def _get_day_zarr_paths(data_dir: Path) -> List[Path]:
    """Return sorted list of day Zarr directories under ocean_met/."""
    ocean_met = data_dir / "ocean_met"
    paths = sorted(p for p in ocean_met.iterdir() if p.suffix == ".zarr")
    if not paths:
        raise FileNotFoundError(f"No .zarr directories found in {ocean_met}")
    return paths


def get_ocean_mask(
    zarr_path: Path,
    time_idx: int,
    coastal_buffer: int = 0,
) -> np.ndarray:
    """Return a 2-D boolean ocean mask (True = ocean pixel) from a zarr file.

    Parameters
    ----------
    zarr_path      : Path to a single day's .zarr directory.
    time_idx       : Half-hourly index in [0, 47].
    coastal_buffer : Pixels to erode from the coastline boundary.
                     0 = no erosion (all ocean pixels kept).
    """
    from scipy.ndimage import binary_erosion
    arr = zarr.open(str(zarr_path), mode="r")
    landsea = np.asarray(arr[MASK_IDX, :, :, time_idx], dtype=np.float32)
    ocean = np.abs(landsea - 1.0) <= 1e-3          # True where mask == 1
    if coastal_buffer > 0:
        ocean = binary_erosion(ocean, iterations=coastal_buffer)
    return ocean


def extract_snapshot(
    zarr_path: Path,
    time_idx: int,
    field_indices: List[int],
    ocean_mask: np.ndarray,
) -> np.ndarray:
    """Load one (N_ocean, C) float64 snapshot from a zarr file.

    Land-bleed NaN/Inf values within the ocean mask are zeroed out.
    """
    arr = zarr.open(str(zarr_path), mode="r")
    rows, cols = np.where(ocean_mask)
    n_ocean = rows.shape[0]
    C = len(field_indices)
    snap = np.empty((n_ocean, C), dtype=np.float64)
    for c, idx in enumerate(field_indices):
        raw = np.asarray(arr[idx, :, :, time_idx], dtype=np.float32)
        vals = raw[rows, cols].astype(np.float64)
        snap[:, c] = np.where(np.isfinite(vals), vals, 0.0)
    return snap


# ---------------------------------------------------------------------------
# NetCDF loading (full-precision OSN files)
# ---------------------------------------------------------------------------

def _get_day_nc_paths(data_dir: Path) -> List[Path]:
    """Return sorted list of daily NetCDF files (YYYYMMDD.nc) in *data_dir*."""
    paths = sorted(data_dir.glob("[0-9]*.nc"))
    if not paths:
        raise FileNotFoundError(
            f"No daily NetCDF files (YYYYMMDD.nc) found in {data_dir}"
        )
    return paths


def _apply_coastal_buffer(ocean: np.ndarray, coastal_buffer: int) -> np.ndarray:
    if coastal_buffer > 0:
        from scipy.ndimage import binary_erosion
        ocean = binary_erosion(ocean, iterations=coastal_buffer)
    return ocean


def get_ocean_mask_nc(
    nc_path: Path,
    static_nc: Path,
    time_idx: int,
    coastal_buffer: int = 0,
) -> np.ndarray:
    """Return a 2-D boolean ocean mask (True = ocean pixel) from NetCDF files.

    The mask source is controlled by NC_MASK_FILE / NC_MASK_VAR at module level.
    *time_idx* is accepted for API symmetry with get_ocean_mask() but is not
    used when the mask comes from static.nc (which is time-invariant).
    """
    if not _XR_AVAILABLE:
        raise ImportError("xarray is required for NetCDF loading: conda install xarray")

    mask_path = static_nc if NC_MASK_FILE == "static" else nc_path
    ds = xr.open_dataset(mask_path, engine=_nc_engine(), mask_and_scale=False)
    if NC_MASK_VAR not in ds:
        available = list(ds.data_vars)
        raise KeyError(
            f"Mask variable '{NC_MASK_VAR}' not found in {mask_path}. "
            f"Available variables: {available}. "
            f"Update NC_MASK_VAR in xspec_coherence.py."
        )
    raw = np.asarray(ds[NC_MASK_VAR].values, dtype=np.float32)
    ds.close()

    # Squeeze any leading singleton dims (e.g. if shape is (1, H, W))
    if raw.ndim > 2:
        raw = raw.squeeze()
    ocean = np.abs(raw - 1.0) <= 1e-3  # True where mask == 1 (ocean)
    return _apply_coastal_buffer(ocean, coastal_buffer)


def extract_snapshot_nc(
    nc_path: Path,
    time_idx: int,
    field_indices: List[int],
    ocean_mask: np.ndarray,
) -> np.ndarray:
    """Load one (N_ocean, C) float64 snapshot from a full-precision NetCDF file.

    Uses NC_FIELD_MAP, NC_TIME_DIM, NC_LEVEL_DIM, IDX_TO_META, and
    FIELD_METADATA from module scope to resolve each field_index to its
    NetCDF variable name and vertical level.

    Land-bleed NaN/Inf values within the ocean mask are zeroed out.
    """
    if not _XR_AVAILABLE:
        raise ImportError("xarray is required for NetCDF loading: conda install xarray")

    ds = xr.open_dataset(nc_path, engine=_nc_engine(), mask_and_scale=False)
    rows, cols = np.where(ocean_mask)
    n_ocean = rows.shape[0]
    C = len(field_indices)
    snap = np.empty((n_ocean, C), dtype=np.float64)

    for c, idx in enumerate(field_indices):
        meta   = IDX_TO_META[idx]
        name   = meta["name"]
        level  = meta["level"]
        is_3d  = FIELD_METADATA[name]["n_levels"] > 1

        nc_var = NC_FIELD_MAP.get(name)
        if nc_var is None:
            raise KeyError(
                f"Field '{name}' not in NC_FIELD_MAP. "
                f"Update NC_FIELD_MAP in xspec_coherence.py after running inspect_nc.py."
            )
        if nc_var not in ds:
            raise KeyError(
                f"Variable '{nc_var}' (mapped from '{name}') not found in {nc_path}. "
                f"Available: {list(ds.data_vars)}. Fix NC_FIELD_MAP."
            )

        var = ds[nc_var]

        # Select time step
        if NC_TIME_DIM in var.dims:
            var = var.isel({NC_TIME_DIM: time_idx})

        # Select vertical level for 3-D fields
        if is_3d:
            if NC_LEVEL_DIM in var.dims:
                var = var.isel({NC_LEVEL_DIM: level})
            else:
                # Try to infer: pick the dim whose size == 12
                level_dim = next(
                    (d for d in var.dims if var.sizes[d] == FIELD_METADATA[name]["n_levels"]),
                    None,
                )
                if level_dim is None:
                    raise ValueError(
                        f"Cannot locate level dimension for 3-D field '{nc_var}' "
                        f"in {nc_path}. dims={var.dims}, shapes={var.shape}. "
                        f"Set NC_LEVEL_DIM in xspec_coherence.py."
                    )
                var = var.isel({level_dim: level})

        raw  = np.asarray(var.values, dtype=np.float64)
        vals = raw[rows, cols]
        snap[:, c] = np.where(np.isfinite(vals), vals, 0.0)

    ds.close()
    return snap


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def extract_coords(ocean_mask: np.ndarray) -> np.ndarray:
    """Return (N_ocean, 2) float64 array of (row, col) grid indices for ocean pixels."""
    rows, cols = np.where(ocean_mask)
    return np.stack([rows, cols], axis=1).astype(np.float64)


# ---------------------------------------------------------------------------
# Unified ensemble loader — auto-detects zarr vs. NetCDF
# ---------------------------------------------------------------------------

def load_coastalbench_ensemble(
    data_dir: Path,
    field_indices: List[int],
    field_labels: List[str],
    ensemble_size: int,
    time_idx: int,
    day_stride: int = 1,
    coastal_buffer: int = 0,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """Load an ensemble of CoastalBench snapshots.

    Automatically detects whether *data_dir* contains the float16 zarr preview
    (HuggingFace layout) or the full-precision NetCDF files (OSN layout) and
    dispatches to the appropriate loading functions.

    One snapshot per day at a fixed half-hourly index. Days are sampled
    uniformly (with stride) across available files up to ``ensemble_size``.

    Returns
    -------
    coords     : (N_ocean, 2)  row/col coordinates for ocean pixels
    snapshots  : list of (N_ocean, C) arrays, one per day
    ocean_mask : (H, W) bool 2-D mask used for this call
    """
    fmt = _detect_data_format(data_dir)

    if fmt == "zarr":
        day_paths = _get_day_zarr_paths(data_dir)
        day_paths = day_paths[::day_stride][:ensemble_size]
        _get_mask = lambda p: get_ocean_mask(p, time_idx, coastal_buffer)
        _get_snap = lambda p: extract_snapshot(p, time_idx, field_indices, ocean_mask)
        fmt_label = "zarr (float16 preview)"
    else:
        static_nc = data_dir / "static.nc"
        if NC_MASK_FILE == "static" and not static_nc.exists():
            raise FileNotFoundError(
                f"static.nc not found in {data_dir}. "
                f"Re-run download_CB_heavy.py — static.nc is always included."
            )
        day_paths = _get_day_nc_paths(data_dir)
        day_paths = day_paths[::day_stride][:ensemble_size]
        _get_mask = lambda p: get_ocean_mask_nc(p, static_nc, time_idx, coastal_buffer)
        _get_snap = lambda p: extract_snapshot_nc(p, time_idx, field_indices, ocean_mask)
        fmt_label = "netcdf (full-precision)"

    if len(day_paths) < ensemble_size and verbose:
        print(
            f"[CB] Warning: only {len(day_paths)} days available "
            f"(requested {ensemble_size} with stride {day_stride})."
        )

    if verbose:
        print(f"[CB] Data format : {fmt_label}")

    ocean_mask = _get_mask(day_paths[0])
    coords     = extract_coords(ocean_mask)
    n_ocean    = coords.shape[0]

    if verbose:
        print(
            f"[CB] Ocean pixels (buffer={coastal_buffer} px): "
            f"{n_ocean:,} / {ocean_mask.size:,} total"
        )
        print(
            f"[CB] Loading {len(day_paths)} snapshots at "
            f"time_idx={time_idx} ..."
        )

    snapshots: List[np.ndarray] = []
    for i, dp in enumerate(day_paths):
        snap = _get_snap(dp)
        snapshots.append(snap)
        if verbose:
            print(f"[CB]   {i + 1:3d}/{len(day_paths)}  {dp.name}")

    return coords, snapshots, ocean_mask


# ============================================================
# New metric computation
# ============================================================
def compute_pearson_matrix(snapshots: List[np.ndarray]) -> np.ndarray:
    """Ensemble-averaged spatial Pearson correlation matrix.

    For each field pair (i, j): ρ_{ij} = mean over snapshots of
    the spatial Pearson correlation between field i and field j.

    Parameters
    ----------
    snapshots : list of (N, C) arrays

    Returns
    -------
    (C, C) symmetric matrix with diagonal = 1, values clipped to [-1, 1].
    """
    C = snapshots[0].shape[1]
    rho_sum = np.zeros((C, C), dtype=np.float64)
    B = len(snapshots)
    for snap in snapshots:
        mu = snap.mean(axis=0)                  # (C,)
        sigma = snap.std(axis=0) + EPS          # (C,)
        z = (snap - mu) / sigma                 # (N, C)  zero-mean, unit-std
        N = z.shape[0]
        rho_sum += (z.T @ z) / N               # (C, C)
    return np.clip(rho_sum / B, -1.0, 1.0)


def compute_band_energy_cv(
    basis: GraphSpectralBasis,
    snapshots: List[np.ndarray],
) -> np.ndarray:
    """Coefficient of variation (CV = std/mean) of band energies across the ensemble.

    High CV means a field's scale content varies meaningfully across days —
    ideal for the cross-spectral loss to learn from.

    Returns
    -------
    (3, C) array of CV values (non-negative), one row per scale band.
    """
    be = compute_band_energies(basis, snapshots)   # BandEnergyEstimate
    raw = be.raw                                   # (B, 3, C)
    mean = raw.mean(axis=0) + EPS                  # (3, C)
    std  = raw.std(axis=0)                         # (3, C)
    return std / mean                              # (3, C)


def compute_temporal_coherence(
    basis: GraphSpectralBasis,
    snapshots: List[np.ndarray],
    n_freq_bins: int = 24,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-field binned coherence between consecutive-day snapshot pairs.

    For each consecutive pair (day_i, day_{i+1}), project both snapshots
    onto the eigenbasis and compute the binned magnitude-squared coherence.
    Results are averaged over all consecutive pairs in the ensemble.

    High temporal coherence (γ² near 1) means consecutive days are nearly
    identical — the ensemble members are too autocorrelated to act as
    independent realizations. Low values indicate fast decorrelation.

    Returns
    -------
    centers : (n_bins,)   graph-frequency bin centers
    gamma2  : (n_bins, C) temporal coherence per field
    """
    bin_assign, centers = frequency_bins(basis, n_freq_bins)
    C = snapshots[0].shape[1]
    n_bins = int(centers.shape[0])
    gamma2_sum = np.zeros((n_bins, C), dtype=np.float64)
    n_pairs_used = 0

    for i in range(len(snapshots) - 1):
        a_i = basis.eigvecs.T @ snapshots[i]       # (K, C)
        a_j = basis.eigvecs.T @ snapshots[i + 1]   # (K, C)
        auto_i = a_i ** 2                           # (K, C)
        auto_j = a_j ** 2
        cross  = a_i * a_j                          # (K, C) real basis

        for b in range(n_bins):
            m = bin_assign == b
            if not m.any():
                continue
            A_i = auto_i[m].sum(axis=0)             # (C,)
            A_j = auto_j[m].sum(axis=0)
            X   = cross[m].sum(axis=0)
            gamma2_sum[b] += X ** 2 / (A_i * A_j + EPS)

        n_pairs_used += 1

    if n_pairs_used == 0:
        return centers, np.zeros((n_bins, C))

    return centers, np.clip(gamma2_sum / n_pairs_used, 0.0, 1.0)


# ============================================================
# Shared figure helpers
# ============================================================
CMAP_COH     = "magma"
CMAP_PEARSON = "RdBu_r"
COL_GT       = "#000000"
COL_BUF      = "#1565C0"     # blue for the coastal-buffer comparison line

_BAND_NAMES = ("large", "medium", "small")


def _shade_bands(ax, band_edges: np.ndarray) -> None:
    """Shade the three scale-band regions on a frequency-axis plot."""
    for b in range(3):
        ax.axvspan(
            band_edges[b], band_edges[b + 1],
            color=BAND_TINT[b], alpha=0.7, lw=0,
        )


def _pair_label(names: Sequence[str], i: int, j: int) -> str:
    return f"{names[i]}\u2013{names[j]}"


def resolve_pairs(
    tokens: Optional[List[str]],
    labels: Sequence[str],
) -> List[Tuple[int, int]]:
    """Parse --pairs tokens ('a,b') into ascending (i, j) index tuples."""
    if not tokens:
        return []
    name_to_c = {n: c for c, n in enumerate(labels)}
    pairs: List[Tuple[int, int]] = []
    for tok in tokens:
        parts = [t.strip() for t in tok.split(",")]
        if len(parts) != 2:
            raise SystemExit(f"--pairs expects 'a,b' tokens; got {tok!r}.")
        resolved = []
        for pname in parts:
            if pname not in name_to_c:
                raise SystemExit(
                    f"Pair token {pname!r} is not among the selected fields "
                    f"{list(labels)}."
                )
            resolved.append(name_to_c[pname])
        i, j = sorted(resolved)
        if i == j:
            raise SystemExit(f"--pairs needs two distinct fields; got {tok!r}.")
        pairs.append((i, j))
    return pairs


# ============================================================
# GT-only figure panels (A – D)
# ============================================================
def draw_coherence_matrix_gt(
    axes: Sequence,
    mat_gt: np.ndarray,    # (3, C, C)
    names: Sequence[str],
) -> object:
    """Single row of 3 band coherence heatmaps (GT-only).

    Parameters
    ----------
    axes   : sequence of 3 Axes objects (e.g. from plt.subplots(1, 3))
    mat_gt : (3, C, C) coherence matrix, values in [0, 1]

    Returns the last imshow handle (for the shared colorbar).
    """
    C = mat_gt.shape[1]
    im = None
    for b, (ax, band_name) in enumerate(zip(axes, _BAND_NAMES)):
        im = ax.imshow(mat_gt[b], cmap=CMAP_COH, vmin=0.0, vmax=1.0, aspect="equal")
        ax.set_xticks(range(C))
        ax.set_yticks(range(C))
        ax.set_xticklabels(names, fontsize=7.0, rotation=45, ha="right")
        ax.set_yticklabels(names if b == 0 else [], fontsize=7.0)
        ax.set_title(f"{band_name} scale", fontsize=9)
        for a in range(C):
            for bb in range(C):
                v = mat_gt[b, a, bb]
                ax.text(
                    bb, a, f"{v:.2f}",
                    ha="center", va="center", fontsize=5.5,
                    color="white" if v < 0.55 else "black",
                )
    return im


def draw_coherence_curves_gt(
    sub,
    basis: GraphSpectralBasis,
    est: SpectralEstimate,
    selected_pairs: Sequence[Tuple[int, int]],
    names: Sequence[str],
    n_freq_bins: int = 24,
    title: Optional[str] = None,
) -> None:
    """Coherence γ²(ν) curves for selected field pairs (GT-only)."""
    bin_assign, centers = frequency_bins(basis, n_freq_bins)
    g = coherence_from_bins(est, bin_assign)        # (n_bins, n_pairs)
    all_pairs = est.pairs

    n_sel = len(selected_pairs)
    ncol = min(3, n_sel) if n_sel else 1
    nrow = int(np.ceil(n_sel / ncol)) if n_sel else 1
    gs = sub.add_gridspec(nrow, ncol, hspace=0.55, wspace=0.38,
                          left=0.08, right=0.97, top=0.88, bottom=0.14)

    band_centers = [
        0.5 * (basis.band_edges[b] + basis.band_edges[b + 1]) for b in range(3)
    ]

    for k, pair in enumerate(selected_pairs):
        row, col = k // ncol, k % ncol
        ax = sub.add_subplot(gs[row, col])
        pair = tuple(pair)
        if pair in all_pairs:
            p_idx = all_pairs.index(pair)
        elif (pair[1], pair[0]) in all_pairs:
            p_idx = all_pairs.index((pair[1], pair[0]))
        else:
            ax.set_visible(False)
            continue
        _shade_bands(ax, basis.band_edges)
        ax.plot(centers, g[:, p_idx], color=COL_GT, lw=1.6)
        ax.set_ylim(0, 1.02)
        ax.set_xlim(centers.min(), centers.max())
        ax.set_title(_pair_label(names, *pair), fontsize=8.5)
        ax.set_xlabel("$\\nu_k = \\sqrt{\\lambda_k}$", fontsize=8)
        ax.set_ylabel("$\\gamma^2$", fontsize=8)
        ax.tick_params(labelsize=6.5)
        for b, nm in enumerate(_BAND_NAMES):
            ax.text(band_centers[b], 0.04, nm, ha="center", fontsize=6.5,
                    color=BAND_EDGE_C[b], style="italic")

    for q in range(n_sel, nrow * ncol):
        sub.add_subplot(gs[q // ncol, q % ncol]).axis("off")

    if title:
        sub.suptitle(title, fontsize=10.5, fontweight="bold", y=0.99)


def draw_auto_spectra_gt(
    sub,
    est: SpectralEstimate,
    basis: GraphSpectralBasis,
    names: Sequence[str],
    n_freq_bins: int = 24,
    title: Optional[str] = None,
) -> None:
    """Log-scale auto-spectrum per field with band shading (GT-only)."""
    bin_assign, centers = frequency_bins(basis, n_freq_bins)
    C = est.auto.shape[1]
    n_bins = centers.shape[0]
    auto_gt = np.zeros((n_bins, C))
    for b in range(n_bins):
        m = bin_assign == b
        if m.any():
            auto_gt[b] = est.auto[m].sum(axis=0)
    auto_gt = np.maximum(auto_gt, EPS)

    ncol = min(4, C)
    nrow = int(np.ceil(C / ncol))
    gs = sub.add_gridspec(nrow, ncol, hspace=0.55, wspace=0.38,
                          left=0.08, right=0.97, top=0.88, bottom=0.10)
    for c in range(C):
        row, col = c // ncol, c % ncol
        ax = sub.add_subplot(gs[row, col])
        _shade_bands(ax, basis.band_edges)
        ax.plot(centers, auto_gt[:, c], color=COL_GT, lw=1.4)
        ax.set_yscale("log")
        ax.set_title(names[c], fontsize=8.5)
        ax.set_xlim(centers.min(), centers.max())
        ax.tick_params(labelsize=6.5)
        if row == nrow - 1:
            ax.set_xlabel("$\\nu_k = \\sqrt{\\lambda_k}$", fontsize=8.0)
        if col == 0:
            ax.set_ylabel("$\\hat{p}_c$  (spectral energy)", fontsize=7.5)

    for q in range(C, nrow * ncol):
        sub.add_subplot(gs[q // ncol, q % ncol]).axis("off")

    if title:
        sub.suptitle(title, fontsize=10.5, fontweight="bold", y=0.99)


def draw_crossband_matrices_gt(
    sub,
    Q_gt: np.ndarray,              # (n_pairs, 3, 3)
    pairs: List[Tuple[int, int]],
    names: Sequence[str],
    n_show: int = 6,
    title: Optional[str] = None,
) -> None:
    """Cross-band coupling Q̂ matrices for the top field pairs (GT-only).

    Pairs are ranked by descending mean off-diagonal GT coupling.
    Diagonal cells (excluded from L_crossfreq) are marked with '*'.
    """
    n_bands = Q_gt.shape[1]
    band_tick = [b[0].upper() for b in _BAND_NAMES]
    offdiag = ~np.eye(n_bands, dtype=bool)

    mean_od = np.array([(Q_gt[p] * offdiag).mean() for p in range(len(pairs))])
    order = np.argsort(-mean_od)
    n_show = min(n_show, len(pairs))
    show = order[:n_show]

    gs = sub.add_gridspec(1, n_show, wspace=0.45,
                          left=0.06, right=0.90, top=0.82, bottom=0.14)
    im = None
    for col, p in enumerate(show):
        c1, c2 = pairs[p]
        ax = sub.add_subplot(gs[0, col])
        im = ax.imshow(Q_gt[p], cmap=CMAP_COH, vmin=0.0, vmax=1.0,
                       aspect="equal", interpolation="nearest")
        ax.set_xticks(range(n_bands))
        ax.set_yticks(range(n_bands))
        ax.set_xticklabels(band_tick, fontsize=6.5)
        ax.set_yticklabels(band_tick if col == 0 else [], fontsize=6.5)
        ax.set_title(_pair_label(names, c1, c2), fontsize=7.5, pad=3)
        if col == 0:
            ax.set_ylabel("Band $B_{c_1}$", fontsize=7.0)
        ax.set_xlabel("Band $B_{c_2}$", fontsize=7.0)
        for bi in range(n_bands):
            for bj in range(n_bands):
                v = Q_gt[p, bi, bj]
                if bi == bj:
                    ax.text(bj, bi, "*", ha="center", va="center",
                            fontsize=7.5, color="gray", zorder=3)
                else:
                    ax.text(bj, bi, f"{v:.2f}", ha="center", va="center",
                            fontsize=6.0,
                            color="white" if v < 0.4 else "black", zorder=3)

    if im is not None:
        cax = sub.add_axes([0.92, 0.20, 0.013, 0.55])
        cb = sub.figure.colorbar(im, cax=cax)
        cb.set_label("$\\hat{q}$ coupling", fontsize=7.5)
        cb.ax.tick_params(labelsize=7.0)

    if title:
        sub.suptitle(title, fontsize=10.5, fontweight="bold", y=0.97)


# ============================================================
# New CB-specific figure panels (E – I)
# ============================================================
def draw_pearson_heatmap(
    sub,
    pearson: np.ndarray,    # (C, C)
    names: Sequence[str],
    title: Optional[str] = None,
) -> None:
    """Ensemble-averaged spatial Pearson correlation heatmap."""
    C = pearson.shape[0]
    gs = sub.add_gridspec(1, 1, left=0.14, right=0.88, top=0.88, bottom=0.14)
    ax = sub.add_subplot(gs[0, 0])
    im = ax.imshow(pearson, cmap=CMAP_PEARSON, vmin=-1.0, vmax=1.0, aspect="equal")
    ax.set_xticks(range(C))
    ax.set_yticks(range(C))
    ax.set_xticklabels(names, fontsize=7.0, rotation=45, ha="right")
    ax.set_yticklabels(names, fontsize=7.0)
    for i in range(C):
        for j in range(C):
            v = pearson[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5,
                    color="white" if abs(v) > 0.6 else "black")
    cax = sub.add_axes([0.90, 0.14, 0.015, 0.74])
    cb = sub.figure.colorbar(im, cax=cax)
    cb.set_label("Pearson $\\rho$", fontsize=8)
    cb.set_ticks([-1, -0.5, 0, 0.5, 1])
    cb.ax.tick_params(labelsize=7.0)
    if title:
        sub.suptitle(title, fontsize=10.5, fontweight="bold", y=0.99)


def draw_pairwise_gamma2_histogram(
    sub,
    est: SpectralEstimate,
    basis: GraphSpectralBasis,
    title: Optional[str] = None,
) -> None:
    """Histogram of band-averaged γ² across all field pairs.

    A dataset well-suited for training with the cross-spectral loss should
    show a broad spread (not collapsed at 0 or 1). Three overlapping
    histograms are shown, one per scale band.
    """
    gb = band_coherence(est, basis)   # (3, n_pairs)
    gs = sub.add_gridspec(1, 1, left=0.12, right=0.97, top=0.85, bottom=0.14)
    ax = sub.add_subplot(gs[0, 0])
    colors = [BAND_EDGE_C[b] for b in range(3)]
    for b in range(3):
        ax.hist(gb[b], bins=20, range=(0, 1), alpha=0.55,
                color=colors[b], label=_BAND_NAMES[b],
                edgecolor="white", lw=0.4)
    ax.axvline(0.1, color="gray", ls="--", lw=0.9, label="low threshold (0.1)")
    ax.axvline(0.9, color="gray", ls=":",  lw=0.9, label="high threshold (0.9)")
    ax.set_xlabel("$\\bar{\\gamma}^2$  (band-averaged per pair)", fontsize=9)
    ax.set_ylabel("Number of field pairs", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper center")
    ax.tick_params(labelsize=7.5)
    if title:
        sub.suptitle(title, fontsize=10.5, fontweight="bold", y=0.99)


def draw_temporal_coherence(
    sub,
    centers: np.ndarray,          # (n_bins,)
    gamma2: np.ndarray,           # (n_bins, C)
    names: Sequence[str],
    basis: GraphSpectralBasis,
    title: Optional[str] = None,
) -> None:
    """Per-field temporal coherence curves (day_i vs day_{i+1}), ensemble-averaged.

    A horizontal dashed line at γ² = 0.5 marks the halfway point.
    High values across all frequencies indicate strong day-to-day autocorrelation.
    """
    C = gamma2.shape[1]
    ncol = min(4, C)
    nrow = int(np.ceil(C / ncol))
    gs = sub.add_gridspec(nrow, ncol, hspace=0.55, wspace=0.38,
                          left=0.08, right=0.97, top=0.88, bottom=0.12)
    for c in range(C):
        row, col = c // ncol, c % ncol
        ax = sub.add_subplot(gs[row, col])
        _shade_bands(ax, basis.band_edges)
        ax.plot(centers, gamma2[:, c], color=COL_GT, lw=1.4)
        ax.axhline(0.5, color="gray", ls="--", lw=0.7)
        ax.set_ylim(0, 1.02)
        ax.set_xlim(centers.min(), centers.max())
        ax.set_title(names[c], fontsize=8.5)
        ax.tick_params(labelsize=6.5)
        if row == nrow - 1:
            ax.set_xlabel("$\\nu_k$", fontsize=8)
        if col == 0:
            ax.set_ylabel("$\\gamma^2_{\\rm temp}$", fontsize=7.5)

    for q in range(C, nrow * ncol):
        sub.add_subplot(gs[q // ncol, q % ncol]).axis("off")

    if title:
        sub.suptitle(title, fontsize=10.5, fontweight="bold", y=0.99)


def draw_band_energy_cv(
    sub,
    cv: np.ndarray,          # (3, C)
    names: Sequence[str],
    title: Optional[str] = None,
) -> None:
    """Grouped bar chart: band energy CV per field per scale band.

    High CV = the field's energy in that band varies substantially across
    ensemble members — good signal for the cross-spectral loss.
    """
    C = cv.shape[1]
    x = np.arange(C)
    width = 0.27
    gs = sub.add_gridspec(1, 1, left=0.08, right=0.97, top=0.84, bottom=0.22)
    ax = sub.add_subplot(gs[0, 0])
    for b in range(3):
        ax.bar(x + (b - 1) * width, cv[b], width=width,
               color=BAND_TINT[b], edgecolor=BAND_EDGE_C[b], lw=0.8,
               label=_BAND_NAMES[b])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=50, ha="right", fontsize=7.0)
    ax.set_ylabel("Band energy CV  (std / mean)", fontsize=8.5)
    ax.legend(fontsize=7.5, title="scale band", title_fontsize=7.0,
              loc="upper right")
    ax.tick_params(axis="y", labelsize=7.5)
    ax.margins(x=0.01)
    if title:
        sub.suptitle(title, fontsize=10.5, fontweight="bold", y=0.99)


def draw_coastal_buffer_comparison(
    sub,
    est_full: SpectralEstimate,
    est_buf: SpectralEstimate,
    basis_full: GraphSpectralBasis,
    basis_buf: GraphSpectralBasis,
    names: Sequence[str],
    n_freq_bins: int = 24,
    coastal_buffer: int = 0,
    title: Optional[str] = None,
) -> None:
    """Auto-spectrum comparison: full ocean mask vs. coastal-buffer-eroded mask.

    Overlays the two auto-spectrum curves for each field. Divergence at
    small scales indicates that near-coastline pixels carry high-frequency
    artefacts from the mask boundary.
    """
    C = est_full.auto.shape[1]

    def _binned_auto(est: SpectralEstimate, basis: GraphSpectralBasis
                     ) -> Tuple[np.ndarray, np.ndarray]:
        bin_assign, centers = frequency_bins(basis, n_freq_bins)
        n_bins = centers.shape[0]
        auto = np.zeros((n_bins, C))
        for b in range(n_bins):
            m = bin_assign == b
            if m.any():
                auto[b] = est.auto[m].sum(axis=0)
        return np.maximum(auto, EPS), centers

    auto_full, centers_full = _binned_auto(est_full, basis_full)
    auto_buf,  centers_buf  = _binned_auto(est_buf,  basis_buf)

    ncol = min(4, C)
    nrow = int(np.ceil(C / ncol))
    gs = sub.add_gridspec(nrow, ncol, hspace=0.55, wspace=0.38,
                          left=0.08, right=0.97, top=0.88, bottom=0.10)
    for c in range(C):
        row, col = c // ncol, c % ncol
        ax = sub.add_subplot(gs[row, col])
        _shade_bands(ax, basis_full.band_edges)
        ax.plot(centers_full, auto_full[:, c], color=COL_GT, lw=1.4,
                label="full ocean" if c == 0 else None)
        ax.plot(centers_buf,  auto_buf[:, c],  color=COL_BUF, lw=1.4, ls="--",
                label=f"buffer={coastal_buffer} px" if c == 0 else None)
        ax.set_yscale("log")
        ax.set_title(names[c], fontsize=8.5)
        ax.set_xlim(centers_full.min(), centers_full.max())
        ax.tick_params(labelsize=6.5)
        if row == nrow - 1:
            ax.set_xlabel("$\\nu_k = \\sqrt{\\lambda_k}$", fontsize=8.0)
        if col == 0:
            ax.set_ylabel("$\\hat{p}_c$  (spectral energy)", fontsize=7.5)
        if c == 0:
            ax.legend(fontsize=7.0, loc="upper right", framealpha=0.85)

    for q in range(C, nrow * ncol):
        sub.add_subplot(gs[q // ncol, q % ncol]).axis("off")

    if title:
        sub.suptitle(title, fontsize=10.5, fontweight="bold", y=0.99)


# ============================================================
# Orchestration
# ============================================================
def make_cb_evaluation_figures(
    out_dir: Path,
    basis: GraphSpectralBasis,
    est: SpectralEstimate,
    snapshots: List[np.ndarray],
    names: Sequence[str],
    selected_pairs: Sequence[Tuple[int, int]],
    n_freq_bins: int = 24,
    coastal_buffer: int = 0,
    est_buf: Optional[SpectralEstimate] = None,
    basis_buf: Optional[GraphSpectralBasis] = None,
    dpi: int = 300,
) -> Dict[str, Path]:
    """Render all evaluation panels and save them to *out_dir*.

    Parameters
    ----------
    selected_pairs : pairs to highlight in Panels B and D. If empty, the top-6
                     most coherent GT pairs are auto-selected.

    Returns
    -------
    Dict of saved file paths keyed by panel name.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ps.set_paper_style()
    saved: Dict[str, Path] = {}
    C = len(names)
    pairs_all = est.pairs
    prefix = "xspec_cb"

    # --- pre-compute shared quantities ---
    gb     = band_coherence(est, basis)                     # (3, n_pairs)
    mat_gt = coherence_matrix(gb, pairs_all, C)             # (3, C, C)
    be     = compute_band_energies(basis, snapshots)
    Q_gt   = cross_band_coupling_scores(be, pairs_all)      # (n_pairs, 3, 3)
    pearson = compute_pearson_matrix(snapshots)             # (C, C)
    cv      = compute_band_energy_cv(basis, snapshots)      # (3, C)
    t_centers, t_gamma2 = compute_temporal_coherence(
        basis, snapshots, n_freq_bins)                      # (n_bins,), (n_bins, C)

    # auto-select top pairs if none provided
    if not selected_pairs:
        ranking = np.argsort(-gb.mean(axis=0))
        n_auto  = min(6, len(pairs_all))
        selected_pairs = [pairs_all[int(k)] for k in ranking[:n_auto]]

    # ----------------------------------------------------------------
    # Panel A : Coherence matrix (GT-only, single row of 3 heatmaps)
    # ----------------------------------------------------------------
    sz = max(3.5 + 0.3 * C, 4.5)
    fig, axes = plt.subplots(1, 3, figsize=(3 * sz, sz + 1.2))
    im = draw_coherence_matrix_gt(axes, mat_gt, names)
    fig.suptitle("a   Cross-field coherence matrix  (per scale band, GT only)",
                 fontsize=10.5, fontweight="bold")
    if im is not None:
        cax = fig.add_axes([0.93, 0.15, 0.012, 0.65])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("$\\gamma^2$ coherence", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    fig.subplots_adjust(left=0.10, right=0.92, top=0.88,
                        bottom=0.16 + 0.015 * C, wspace=0.28)
    p = out_dir / f"{prefix}_panelA_coherence_matrix.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    saved["panelA"] = p
    print(f"[CB] panelA -> {p.name}")

    # ----------------------------------------------------------------
    # Panel B : Coherence curves for selected pairs
    # ----------------------------------------------------------------
    n_sel  = min(6, len(selected_pairs))
    ncol_b = min(3, n_sel) if n_sel else 1
    nrow_b = int(np.ceil(n_sel / ncol_b)) if n_sel else 1
    fb = plt.figure(figsize=(4.5 * ncol_b, 3.2 * nrow_b))
    draw_coherence_curves_gt(
        fb.subfigures(1, 1), basis, est,
        selected_pairs[:n_sel], names, n_freq_bins,
        title="b   Cross-spectral coherence curves  $\\gamma^2(\\nu)$  (GT only)",
    )
    p = out_dir / f"{prefix}_panelB_coherence_curves.png"
    fb.savefig(p, dpi=dpi, bbox_inches="tight"); plt.close(fb)
    saved["panelB"] = p
    print(f"[CB] panelB -> {p.name}")

    # ----------------------------------------------------------------
    # Panel C : Auto-spectrum (log scale, per field)
    # ----------------------------------------------------------------
    ncol_c = min(4, C)
    nrow_c = int(np.ceil(C / ncol_c))
    fc = plt.figure(figsize=(3.0 * ncol_c, 2.5 * nrow_c))
    draw_auto_spectra_gt(
        fc.subfigures(1, 1), est, basis, names, n_freq_bins,
        title="c   Per-field auto-spectrum  $\\hat{p}_c[k]$  (GT only, log scale)",
    )
    p = out_dir / f"{prefix}_panelC_auto_spectra.png"
    fc.savefig(p, dpi=dpi, bbox_inches="tight"); plt.close(fc)
    saved["panelC"] = p
    print(f"[CB] panelC -> {p.name}")

    # ----------------------------------------------------------------
    # Panel D : Cross-band coupling matrices (top pairs, GT-only)
    # ----------------------------------------------------------------
    n_show_d = min(6, len(pairs_all))
    fd = plt.figure(figsize=(max(6.0, 2.8 * n_show_d), 4.5))
    draw_crossband_matrices_gt(
        fd.subfigures(1, 1), Q_gt, pairs_all, names, n_show=n_show_d,
        title="d   Cross-band coupling matrices  $\\hat{Q}_{c_1 c_2}^{B_i B_j}$  (GT only)",
    )
    p = out_dir / f"{prefix}_panelD_crossband_matrices.png"
    fd.savefig(p, dpi=dpi, bbox_inches="tight"); plt.close(fd)
    saved["panelD"] = p
    print(f"[CB] panelD -> {p.name}")

    # ----------------------------------------------------------------
    # Panel E : Pearson correlation heatmap
    # ----------------------------------------------------------------
    sz_e = max(4.5, 0.55 * C + 1.5)
    fe = plt.figure(figsize=(sz_e + 1.0, sz_e))
    draw_pearson_heatmap(
        fe.subfigures(1, 1), pearson, names,
        title="e   Ensemble-averaged spatial Pearson correlation",
    )
    p = out_dir / f"{prefix}_panelE_pearson.png"
    fe.savefig(p, dpi=dpi, bbox_inches="tight"); plt.close(fe)
    saved["panelE"] = p
    print(f"[CB] panelE -> {p.name}")

    # ----------------------------------------------------------------
    # Panel F : Pairwise γ² histogram
    # ----------------------------------------------------------------
    ff = plt.figure(figsize=(7.0, 4.0))
    draw_pairwise_gamma2_histogram(
        ff.subfigures(1, 1), est, basis,
        title="f   Distribution of pairwise $\\bar{\\gamma}^2$ (by scale band)",
    )
    p = out_dir / f"{prefix}_panelF_gamma2_histogram.png"
    ff.savefig(p, dpi=dpi, bbox_inches="tight"); plt.close(ff)
    saved["panelF"] = p
    print(f"[CB] panelF -> {p.name}")

    # ----------------------------------------------------------------
    # Panel G : Temporal coherence curves
    # ----------------------------------------------------------------
    ncol_g = min(4, C)
    nrow_g = int(np.ceil(C / ncol_g))
    fg = plt.figure(figsize=(3.0 * ncol_g, 2.5 * nrow_g))
    draw_temporal_coherence(
        fg.subfigures(1, 1), t_centers, t_gamma2, names, basis,
        title="g   Temporal coherence: consecutive-day self-coherence per field",
    )
    p = out_dir / f"{prefix}_panelG_temporal_coherence.png"
    fg.savefig(p, dpi=dpi, bbox_inches="tight"); plt.close(fg)
    saved["panelG"] = p
    print(f"[CB] panelG -> {p.name}")

    # ----------------------------------------------------------------
    # Panel H : Band energy CV bar chart
    # ----------------------------------------------------------------
    fh = plt.figure(figsize=(max(7.0, 0.8 * C), 4.0))
    draw_band_energy_cv(
        fh.subfigures(1, 1), cv, names,
        title="h   Band energy coefficient of variation across the ensemble",
    )
    p = out_dir / f"{prefix}_panelH_band_energy_cv.png"
    fh.savefig(p, dpi=dpi, bbox_inches="tight"); plt.close(fh)
    saved["panelH"] = p
    print(f"[CB] panelH -> {p.name}")

    # ----------------------------------------------------------------
    # Panel I : Coastal buffer comparison (optional)
    # ----------------------------------------------------------------
    if coastal_buffer > 0 and est_buf is not None and basis_buf is not None:
        ncol_i = min(4, C)
        nrow_i = int(np.ceil(C / ncol_i))
        fi = plt.figure(figsize=(3.0 * ncol_i, 2.5 * nrow_i))
        draw_coastal_buffer_comparison(
            fi.subfigures(1, 1),
            est_full=est, est_buf=est_buf,
            basis_full=basis, basis_buf=basis_buf,
            names=names, n_freq_bins=n_freq_bins,
            coastal_buffer=coastal_buffer,
            title=(
                f"i   Auto-spectrum: full ocean vs. "
                f"{coastal_buffer}-pixel coastal buffer"
            ),
        )
        p = out_dir / f"{prefix}_panelI_coastal_buffer.png"
        fi.savefig(p, dpi=dpi, bbox_inches="tight"); plt.close(fi)
        saved["panelI"] = p
        print(f"[CB] panelI -> {p.name}")

    return saved


def write_summary(
    path: Path,
    names: Sequence[str],
    pairs: List[Tuple[int, int]],
    gb: np.ndarray,
    pearson: np.ndarray,
    meta: dict,
) -> None:
    """Write a JSON summary of the key evaluation statistics."""
    band_names_list = list(_BAND_NAMES)
    pair_labels = [_pair_label(names, i, j) for i, j in pairs]
    n_pairs = len(pairs)

    # Top-10 pairs by mean band-averaged γ²
    top_k = min(10, n_pairs)
    top_order = np.argsort(-gb.mean(axis=0))[:top_k]

    summary = {
        "meta": meta,
        "field_names": list(names),
        "n_pairs": n_pairs,
        "band_coherence_stats": {
            band_names_list[b]: {
                "mean": float(gb[b].mean()),
                "std":  float(gb[b].std()),
                "min":  float(gb[b].min()),
                "max":  float(gb[b].max()),
            }
            for b in range(3)
        },
        "pearson_offdiag_mean": float(
            pearson[~np.eye(len(names), dtype=bool)].mean()
        ),
        "top_pairs_by_coherence": {
            pair_labels[int(k)]: {
                band_names_list[b]: float(gb[b, int(k)]) for b in range(3)
            }
            for k in top_order
        },
    }
    path.write_text(json.dumps(summary, indent=2))
    print(f"[CB] summary JSON -> {path}")


# ============================================================
# Main
# ============================================================
def main() -> None:
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---- --list-fields shortcut ----
    if args.list_fields:
        print("Valid CoastalBench field labels:\n")
        print(f"  {'Label':<16s} {'Zarr idx':>8s}  Variable  Level")
        print("  " + "-" * 48)
        for label in sorted(_LABEL_TO_IDX):
            idx = _LABEL_TO_IDX[label]
            m = IDX_TO_META[idx]
            print(
                f"  {label:<16s} {idx:>8d}  {m['name']:<8s}  "
                f"{'—' if m['n_levels'] == 1 else m['level']}"
            )
        return

    # ---- default field selection (surface dynamics + thermodynamics) ----
    if args.fields is None:
        args.fields = [
            "u_z0", "v_z0", "zeta",
            "temp_z0", "Tair", "swrad",
            "Uwind", "Vwind",
        ]
        print(f"[CB] --fields not specified; using defaults: {args.fields}")

    field_indices, field_labels = parse_fields(args.fields)
    C = len(field_labels)
    print(f"[CB] Evaluating {C} field(s): {field_labels}")

    # ---- output directory ----
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = args.out_dir or (repo_root / "figures" / "xspec_eval" / ts)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[CB] Output directory: {out_dir}")

    # ---- data loading (always with full ocean mask) ----
    coords, snapshots, ocean_mask = load_coastalbench_ensemble(
        data_dir=Path(args.data),
        field_indices=field_indices,
        field_labels=field_labels,
        ensemble_size=args.ensemble_size,
        time_idx=args.time,
        day_stride=args.day_stride,
        coastal_buffer=0,       # full mask; buffered pass below if requested
        verbose=True,
    )
    B = len(snapshots)

    # ---- graph-spectral basis (full mask, cached) ----
    cache_full = str(
        args.basis_cache or (out_dir.parent / "graph_basis_cache_full.npz")
    )
    print(
        f"[CB] Computing graph-spectral basis  "
        f"(N={coords.shape[0]:,}, modes={args.n_modes}, k={args.k_neighbors}) ..."
    )
    basis = compute_graph_basis(
        coords,
        n_modes=args.n_modes,
        k_neighbors=args.k_neighbors,
        cache_path=cache_full,
        verbose=True,
    )

    # ---- ensemble cross-spectra ----
    print(f"[CB] Accumulating cross-spectra over {B} snapshots ...")
    est = accumulate_spectra(basis, snapshots)

    # ---- coastal buffer — optional second pass ----
    est_buf = basis_buf = None
    if args.coastal_buffer > 0:
        print(
            f"[CB] Coastal-buffer pass  "
            f"(eroding {args.coastal_buffer} px from coastline) ..."
        )
        coords_buf, snaps_buf, _ = load_coastalbench_ensemble(
            data_dir=Path(args.data),
            field_indices=field_indices,
            field_labels=field_labels,
            ensemble_size=args.ensemble_size,
            time_idx=args.time,
            day_stride=args.day_stride,
            coastal_buffer=args.coastal_buffer,
            verbose=True,
        )
        cache_buf = str(
            out_dir.parent
            / f"graph_basis_cache_buf{args.coastal_buffer}.npz"
        )
        basis_buf = compute_graph_basis(
            coords_buf,
            n_modes=args.n_modes,
            k_neighbors=args.k_neighbors,
            cache_path=cache_buf,
            verbose=True,
        )
        est_buf = accumulate_spectra(basis_buf, snaps_buf)

    # ---- pair resolution ----
    selected_pairs = resolve_pairs(args.pairs, field_labels)

    # ---- figure generation ----
    print("[CB] Rendering evaluation panels ...")
    saved = make_cb_evaluation_figures(
        out_dir=out_dir,
        basis=basis,
        est=est,
        snapshots=snapshots,
        names=field_labels,
        selected_pairs=selected_pairs,
        n_freq_bins=args.n_freq_bins,
        coastal_buffer=args.coastal_buffer,
        est_buf=est_buf,
        basis_buf=basis_buf,
        dpi=300,
    )

    # ---- summary JSON ----
    gb      = band_coherence(est, basis)
    pearson = compute_pearson_matrix(snapshots)
    meta = {
        "data":           str(args.data),
        "fields":         field_labels,
        "ensemble_size":  B,
        "time_idx":       args.time,
        "day_stride":     args.day_stride,
        "n_modes":        args.n_modes,
        "k_neighbors":    args.k_neighbors,
        "coastal_buffer": args.coastal_buffer,
        "timestamp":      ts,
    }
    write_summary(
        out_dir / "xspec_cb_summary.json",
        field_labels, est.pairs, gb, pearson, meta,
    )

    print(f"\n[CB] Done. {len(saved)} panel(s) saved to {out_dir}")
    for k, v in saved.items():
        print(f"  {k:<14s}  {v.name}")


if __name__ == "__main__":
    main()
