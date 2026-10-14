"""
hdf5convert.py — Convert carved CoastalBench NetCDF subsets to HDF5.

Reads the output of ``carve_cb_heavy.py`` (a directory of daily YYYYMMDD.nc
files plus static.nc) and writes a single HDF5 file that conforms to the
canonical ``[B,T,Nx,Ny,Nz,C]`` schema consumed by H5FieldDataset.

Ocean masking
-------------
``mask_rho`` from ``static.nc`` (1 = ocean, 0 = land) is used to drop all
land grid points.  Only the N_ocean wet points are retained, giving a
flattened irregular point-cloud layout::

    /fields      [B, T, N_ocean, 1, 1, C]   float32
    /coordinates [N_ocean, 1, 1, 3]         float32   (lon, lat, z=0)

This exactly mirrors the original CoastalBenchNCDataset masking logic.

Trajectory layout
-----------------
Each carved date becomes one independent trajectory (B = N_dates).
T = number of ocean_time steps per day (typically 48 for 30-min output),
optionally subsampled by ``--time-stride N`` so T_stored = ceil(T/N).
The first 80 % of dates are written to splits/train, the next 10 % to
splits/validation, and the last 10 % to splits/test.

Incremental write (memory-safe)
--------------------------------
Fields are written one day at a time directly into a pre-allocated HDF5
dataset; the entire dataset is NEVER held in RAM simultaneously.  This
eliminates the double-memory spike of the old accumulate-then-stack approach
and allows arbitrarily large datasets to be converted on any node with modest
memory (~1–2 GB typical peak).

Normalization statistics (mean/std, p1/p99) are computed online over training
trajectories only, using Welford's parallel algorithm and reservoir sampling —
no second pass or re-read required.

Time stride
-----------
Two independent time-stride controls exist:

1. **Conversion-time** ``--time-stride N`` (this script):
   Stores every Nth snapshot in the HDF5.  Reduces disk footprint permanently.
   T_stored = len(range(0, T_per_day, N)).

2. **Training-time** ``time_stride`` in ``dataset.yaml``:
   Sub-selects from whatever is stored in the HDF5 at load time.
   No re-conversion needed; change the YAML and re-train.

Output directory naming
-----------------------
A new subdirectory is created beside the input (or inside ``--pool-dir``)::

    <pool-dir>/
      cb_tuvz_L11_12d_2008monthly/       ← NC subset (input from carve_cb_heavy.py)
      cb_tuvz_L11_12d_2008monthly_h5/    ← HDF5 output (this script)
        coastal_bench.h5
        convert_manifest.json
        convert_info.yaml

Usage
-----
  # Incremental conversion (default, memory-safe)
  python hdf5convert.py \\
      --input-dir /path/to/pool/cb_tuvz_L11_12d_2008monthly \\
      --pool-dir  /path/to/pool

  # With time stride: store every 2nd snapshot (24 per day instead of 48)
  python hdf5convert.py \\
      --input-dir /path/to/pool/cb_dstuv_L0_730d_2009_2010daily \\
      --pool-dir  /path/to/pool \\
      --time-stride 2

  # Force re-conversion
  python hdf5convert.py \\
      --input-dir /path/to/pool/cb_tuvz_L11_12d_2008monthly \\
      --force

Notes
-----
  * Requires h5py, netCDF4, numpy, pyyaml (h5py is a core dep of this repo).
  * Statistics are computed on train trajectories only using online algorithms.
  * Re-running is safe: already-converted output is skipped unless --force.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Variable sets (must stay in sync with carve_cb_heavy.py)
# ---------------------------------------------------------------------------
CB_3D_VARS = frozenset({"u", "v", "w", "temp", "salt", "AKs"})
CB_2D_VARS = frozenset({"zeta", "Pair", "Qair", "Tair", "rain",
                         "swrad", "lwrad", "Uwind", "Vwind",
                         "wetdry_mask_rho"})
ALL_CB_VARS = CB_3D_VARS | CB_2D_VARS

DEFAULT_VARS  = ["u", "v", "temp", "zeta"]
DEFAULT_LEVEL = 11   # ROMS surface (0 = bottom in 12-level model)

# Output HDF5 filename inside the output directory
_HDF5_FILENAME     = "coastal_bench.h5"
_MANIFEST_FILENAME = "convert_manifest.json"
_INFO_FILENAME     = "convert_info.yaml"

# Reservoir sampler: max samples per channel for percentile estimation
_RESERVOIR_SIZE = 2_000_000

# ---------------------------------------------------------------------------
# Static-file helpers (ocean mask)
# ---------------------------------------------------------------------------

def _load_static(
    static_nc: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read the ocean mask and grid coordinates from static.nc.

    Parameters
    ----------
    static_nc : str
        Path to static.nc (copied into the carved NC subset directory).

    Returns
    -------
    ocean_mask : ndarray, shape (eta, xi), dtype bool
    lon_full : ndarray, shape (eta, xi), dtype float32
    lat_full : ndarray, shape (eta, xi), dtype float32
    ocean_flat_idx : ndarray, shape (N_ocean,), dtype int64
    lon_ocean : ndarray, shape (N_ocean,), dtype float32
    lat_ocean : ndarray, shape (N_ocean,), dtype float32
    """
    import netCDF4 as nc  # lazy import — not a core repo dep

    with nc.Dataset(static_nc, "r") as ds:
        if "mask_rho" not in ds.variables:
            raise KeyError(f"'mask_rho' not found in {static_nc}. "
                           "Is this a valid CoastalBench static.nc?")
        mask_rho = ds.variables["mask_rho"][:].data  # (eta, xi)
        if "lon_rho" in ds.variables:
            lon = ds.variables["lon_rho"][:].data
            lat = ds.variables["lat_rho"][:].data
        elif "lon" in ds.variables:
            lon = ds.variables["lon"][:].data
            lat = ds.variables["lat"][:].data
        else:
            raise KeyError(f"Neither 'lon_rho' nor 'lon' found in {static_nc}.")

    lon_full = lon.astype(np.float32)
    lat_full = lat.astype(np.float32)

    ocean_mask = (mask_rho == 1)
    ocean_flat_idx = np.flatnonzero(ocean_mask).astype(np.int64)
    lon_ocean = lon_full.ravel()[ocean_flat_idx]
    lat_ocean = lat_full.ravel()[ocean_flat_idx]

    n_total  = ocean_mask.size
    n_ocean  = ocean_flat_idx.size
    pct      = 100.0 * n_ocean / max(n_total, 1)
    print(f"  ocean mask : {n_ocean} / {n_total} points ({pct:.1f} % wet)")
    return ocean_mask, lon_full, lat_full, ocean_flat_idx, lon_ocean, lat_ocean


# ---------------------------------------------------------------------------
# Bounding-box crop helper
# ---------------------------------------------------------------------------

def _apply_bbox_mask(
    ocean_mask: np.ndarray,      # (eta, xi) bool
    lon_full: np.ndarray,        # (eta, xi) float32 — full grid
    lat_full: np.ndarray,        # (eta, xi) float32 — full grid
    bbox: Tuple[float, float, float, float],  # (lon_min, lon_max, lat_min, lat_max)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Restrict the ocean point set to a geographic bounding box.
    """
    lon_min, lon_max, lat_min, lat_max = bbox
    bbox_mask = (
        (lon_full >= lon_min) & (lon_full <= lon_max) &
        (lat_full >= lat_min) & (lat_full <= lat_max)
    )
    combined  = ocean_mask & bbox_mask
    flat_idx  = np.flatnonzero(combined).astype(np.int64)
    lon_crop  = lon_full.ravel()[flat_idx]
    lat_crop  = lat_full.ravel()[flat_idx]

    n_ocean  = int(ocean_mask.sum())
    n_crop   = flat_idx.size
    pct      = 100.0 * n_crop / max(n_ocean, 1)
    print(f"  bbox crop  : {n_crop} / {n_ocean} ocean points kept ({pct:.1f} %) "
          f"[lon {lon_min:.2f}..{lon_max:.2f}, lat {lat_min:.2f}..{lat_max:.2f}]")
    if n_crop == 0:
        import warnings
        warnings.warn(
            f"Bounding box {bbox} contains zero ocean points. "
            "Check coordinate sign conventions (CB uses negative west longitudes).",
            stacklevel=3,
        )
    return flat_idx, lon_crop, lat_crop


# ---------------------------------------------------------------------------
# Grid-index bounding-box crop helper
# ---------------------------------------------------------------------------

def _apply_grid_bbox_mask(
    ocean_mask: np.ndarray,   # (eta, xi) bool
    lon_full: np.ndarray,     # (eta, xi) float32
    lat_full: np.ndarray,     # (eta, xi) float32
    eta_min: int,
    eta_max: int,
    xi_min: int,
    xi_max: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Restrict the ocean point set to a rectangular ROMS grid-index window.

    Unlike the lon/lat bbox, this selection is axis-aligned in (eta, xi) space,
    so the resulting points form a perfect rectangle in the 2D training grid
    array (appears as a slanted parallelogram on the geographic map).
    """
    n_eta, n_xi = ocean_mask.shape

    eta_idx = np.arange(n_eta)[:, np.newaxis]   # (eta, 1)
    xi_idx  = np.arange(n_xi)[np.newaxis, :]    # (1, xi)
    window_mask = (
        (eta_idx >= eta_min) & (eta_idx <= eta_max) &
        (xi_idx  >= xi_min)  & (xi_idx  <= xi_max)
    )  # (eta, xi) bool

    combined = ocean_mask & window_mask
    flat_idx = np.flatnonzero(combined).astype(np.int64)
    lon_crop = lon_full.ravel()[flat_idx]
    lat_crop = lat_full.ravel()[flat_idx]

    n_ocean = int(ocean_mask.sum())
    n_crop  = flat_idx.size
    pct     = 100.0 * n_crop / max(n_ocean, 1)
    print(f"  grid bbox  : {n_crop} / {n_ocean} ocean points kept ({pct:.1f} %) "
          f"[eta {eta_min}..{eta_max}, xi {xi_min}..{xi_max}]")
    if n_crop == 0:
        import warnings
        warnings.warn(
            f"Grid bbox eta[{eta_min}..{eta_max}] xi[{xi_min}..{xi_max}] "
            "contains zero ocean points.  Check that the indices are within "
            f"the grid dimensions ({n_eta}×{n_xi}).",
            stacklevel=3,
        )
    return flat_idx, lon_crop, lat_crop


# ---------------------------------------------------------------------------
# Always-wet mask builder
# ---------------------------------------------------------------------------

def _build_persistent_wet_mask(
    dates: List[str],
    input_dir: str,
    static_mask: np.ndarray,   # (eta, xi) bool
) -> np.ndarray:
    """
    Compute the intersection of mask_rho and wetdry_mask_rho across all dates.

    A point is included only if it is wet (wetdry_mask_rho == 1) in every
    carved daily file.  This eliminates tidal flats, barrier islands, and any
    cell that goes dry at any point in the training dataset.
    """
    import netCDF4 as nc  # noqa: PLC0415
    import warnings

    persistent_wet = static_mask.copy()
    missing: List[str] = []

    print(f"  pre-pass   : reading wetdry_mask_rho from {len(dates)} date(s) …",
          flush=True)
    for date in dates:
        nc_path = os.path.join(input_dir, f"{date}.nc")
        if not os.path.exists(nc_path):
            continue
        with nc.Dataset(nc_path, "r") as ds:
            if "wetdry_mask_rho" not in ds.variables:
                missing.append(date)
                continue
            wd = np.array(ds.variables["wetdry_mask_rho"][:], dtype=np.float32)
            if wd.ndim == 3:
                # (ocean_time, eta, xi) → AND across time steps
                wd = wd.min(axis=0)
            persistent_wet &= (wd == 1)

    if missing:
        warnings.warn(
            f"wetdry_mask_rho absent from {len(missing)} carved file(s); "
            "those dates were skipped in the always-wet pre-pass. "
            "Re-carve with --vars ... wetdry_mask_rho to include them.",
            stacklevel=2,
        )

    n_static = int(static_mask.sum())
    n_persistent = int(persistent_wet.sum())
    pct = 100.0 * n_persistent / max(n_static, 1)
    print(f"  always-wet : {n_persistent} / {n_static} static ocean points kept "
          f"({pct:.1f} %)  — {n_static - n_persistent} tidal/intertidal points removed")
    return persistent_wet


# ---------------------------------------------------------------------------
# Daily NetCDF reader
# ---------------------------------------------------------------------------

def _read_daily(
    nc_path: str,
    var_names: Sequence[str],
    ocean_flat_idx: np.ndarray,
    grid_shape: Tuple[int, int],  # (eta, xi)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read one carved daily NetCDF file and apply the ocean mask.

    Returns
    -------
    time_values : ndarray, shape (T_day,), dtype float64
    fields : ndarray, shape (T_day, N_ocean, C), dtype float32
    """
    import netCDF4 as nc

    with nc.Dataset(nc_path, "r") as ds:
        if "ocean_time" not in ds.variables:
            raise KeyError(f"'ocean_time' not found in {nc_path}.")
        t_var = ds.variables["ocean_time"]
        time_raw = np.array(t_var[:], dtype=np.float64)  # (T_day,)

        T_day = time_raw.shape[0]
        N_ocean = ocean_flat_idx.shape[0]
        C = len(var_names)
        eta, xi = grid_shape

        fields = np.empty((T_day, N_ocean, C), dtype=np.float32)

        for c_idx, vname in enumerate(var_names):
            if vname not in ds.variables:
                print(f"    [warn] variable '{vname}' missing in {nc_path}, filling with NaN.")
                fields[:, :, c_idx] = np.nan
                continue

            v = ds.variables[vname]
            data = np.array(v[:], dtype=np.float32)

            if data.ndim == 3:
                # (T_day, eta, xi) — standard carved shape
                if data.shape[1:] != (eta, xi):
                    raise ValueError(
                        f"Spatial dims of '{vname}' in {nc_path} are "
                        f"{data.shape[1:]} but static.nc implies ({eta}, {xi})."
                    )
                flat = data.reshape(T_day, eta * xi)
                fields[:, :, c_idx] = flat[:, ocean_flat_idx]

            elif data.ndim == 2:
                # (eta, xi) — time-independent static field; broadcast over time
                flat = data.ravel()[ocean_flat_idx]            # (N_ocean,)
                fields[:, :, c_idx] = flat[np.newaxis, :]     # broadcast T_day

            else:
                raise ValueError(
                    f"Unexpected ndim={data.ndim} for variable '{vname}' in {nc_path}."
                )

            # Replace masked / fill values with NaN
            fill = getattr(v, "_FillValue", None)
            if fill is not None:
                try:
                    fill_f32 = np.float32(fill)
                    bad = np.abs(fields[:, :, c_idx] - fill_f32) < 1e-3 * np.abs(fill_f32 + 1e-30)
                    fields[:, :, c_idx][bad] = np.nan
                except (TypeError, ValueError):
                    pass

    return time_raw, fields


# ---------------------------------------------------------------------------
# Probe helper — determine T_per_day without loading field data
# ---------------------------------------------------------------------------

def _probe_one_date(
    dates: List[str],
    input_dir: str,
) -> int:
    """
    Open the first available carved date and return T_per_day (the number of
    ocean_time steps per day).  Only the time dimension is read — no field data
    is loaded, so this is very fast regardless of domain size.
    """
    import netCDF4 as nc

    for date in dates:
        nc_path = os.path.join(input_dir, f"{date}.nc")
        if not os.path.exists(nc_path):
            continue
        try:
            with nc.Dataset(nc_path, "r") as ds:
                if "ocean_time" not in ds.variables:
                    continue
                T = int(ds.variables["ocean_time"].shape[0])
                print(f"  probe      : {date}.nc → T_per_day = {T}")
                return T
        except Exception:
            continue
    raise RuntimeError(
        f"Could not probe any date in {input_dir} to determine T_per_day. "
        "Check that YYYYMMDD.nc files exist and contain 'ocean_time'."
    )


# ---------------------------------------------------------------------------
# Online statistics: Welford's algorithm (mean / std, O(1) memory per channel)
# ---------------------------------------------------------------------------

class _WelfordAccumulator:
    """
    Online mean and variance using the parallel (Chan et al.) batch algorithm.

    Each ``update()`` call merges a batch of values into the running totals
    without a per-element Python loop.  NaN values are excluded per channel.

    Memory: O(C) — independent of the number of samples.
    """

    def __init__(self, C: int) -> None:
        self._C   = C
        self.n    = np.zeros(C, dtype=np.float64)   # sample count per channel
        self.mean = np.zeros(C, dtype=np.float64)
        self.M2   = np.zeros(C, dtype=np.float64)   # sum of squared deviations

    def update(self, data: np.ndarray) -> None:
        """
        Merge a batch of samples into the running accumulators.

        Parameters
        ----------
        data : ndarray, shape (..., C), dtype float32
            Any leading dimensions are flattened.  NaN / Inf are excluded.
        """
        data_flat = data.reshape(-1, self._C)   # float32 view (no copy)
        for c in range(self._C):
            # Convert one channel at a time to float64 to limit peak RAM
            col = data_flat[:, c].astype(np.float64)
            col = col[np.isfinite(col)]
            n_b = col.size
            if n_b == 0:
                continue
            mean_b = float(col.mean())
            M2_b   = float(col.var() * n_b)    # sum of sq. deviations for batch

            n_a    = self.n[c]
            mean_a = self.mean[c]
            M2_a   = self.M2[c]
            n_ab   = n_a + n_b
            delta  = mean_b - mean_a

            self.mean[c] = (n_a * mean_a + n_b * mean_b) / n_ab
            self.M2[c]   = M2_a + M2_b + delta**2 * n_a * n_b / n_ab
            self.n[c]    = n_ab

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) as float32 arrays of shape (C,)."""
        mean = self.mean.astype(np.float32)
        var  = np.where(self.n > 1, self.M2 / (self.n - 1), 0.0)
        std  = np.sqrt(var).astype(np.float32)
        return mean, std


# ---------------------------------------------------------------------------
# Online statistics: reservoir sampler (percentile approximation)
# ---------------------------------------------------------------------------

class _ReservoirSampler:
    """
    Fixed-size reservoir sampler for per-channel p1 / p99 estimation.

    Uses Vitter's Algorithm R adapted for vectorised batch updates:
    — while the reservoir has space, incoming values fill it directly;
    — once full, each new value replaces a uniformly random reservoir slot
      with probability ``size / (total_seen + 1)``.

    At the end, ``finalize()`` sorts the reservoir and computes median and
    the (p99 - p1) range used by the 'robust_99' normaliser.

    Memory: O(size × C) ≈ 32 MB for size=2M, C=4 (float32).
    """

    def __init__(self, C: int, size: int = _RESERVOIR_SIZE, seed: int = 42) -> None:
        self._C    = C
        self._size = size
        # Pre-allocated buffer; unfilled slots hold NaN (never used in finalize)
        self._buf    = np.full((size, C), np.nan, dtype=np.float32)
        self._filled = np.zeros(C, dtype=np.int64)   # filled slots per channel
        self._total  = np.zeros(C, dtype=np.int64)   # total valid samples seen
        self._rng    = np.random.default_rng(seed)

    def update(self, data: np.ndarray) -> None:
        """
        Add a batch to the reservoir.

        Parameters
        ----------
        data : ndarray, shape (..., C)
            NaN / Inf values are excluded per channel.
        """
        flat = data.reshape(-1, self._C)   # float32 view
        for c in range(self._C):
            col = flat[:, c]
            col = col[np.isfinite(col)]    # (N_valid,) float32 copy
            if col.size == 0:
                continue

            n_prev     = int(self._total[c])
            n_incoming = col.size
            self._total[c] += n_incoming

            n_filled = int(self._filled[c])
            n_space  = self._size - n_filled

            if n_space > 0:
                # Reservoir not yet full — fill directly
                n_direct = min(n_incoming, n_space)
                self._buf[n_filled:n_filled + n_direct, c] = col[:n_direct]
                self._filled[c] += n_direct
                remaining = col[n_direct:]
                if remaining.size == 0:
                    continue
                # Switch to reservoir-replacement for the leftover samples
                col      = remaining
                n_offset = n_prev + n_direct  # global count before this sub-batch
            else:
                n_offset = n_prev

            # Reservoir is full — vectorised Algorithm R replacement
            n_rem = col.size
            if n_rem == 0:
                continue
            # Acceptance probability for j-th sample: size / (n_offset + j + 1)
            indices = np.arange(n_offset + 1, n_offset + n_rem + 1, dtype=np.float64)
            accept  = self._rng.random(n_rem) < (self._size / indices)
            accepted_vals = col[accept]
            if accepted_vals.size > 0:
                replace_slots = self._rng.integers(0, self._size, size=accepted_vals.size)
                self._buf[replace_slots, c] = accepted_vals

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute per-channel median (offset) and p99-p1 range (scale_99).

        Returns
        -------
        offset : (C,) float32  — per-channel median
        scale  : (C,) float32  — per-channel (p99 − p1) range
        """
        offset = np.zeros(self._C, dtype=np.float32)
        scale  = np.ones(self._C,  dtype=np.float32)

        for c in range(self._C):
            n_fill = int(self._filled[c])
            if n_fill == 0:
                continue
            col = self._buf[:n_fill, c]
            col = col[np.isfinite(col)]
            if col.size == 0:
                continue
            offset[c] = float(np.median(col))
            p1, p99   = np.percentile(col, [1.0, 99.0])
            scale[c]  = float(max(p99 - p1, 1e-8))

        return offset, scale


# ---------------------------------------------------------------------------
# Statistics helpers (legacy path — kept for direct / small-dataset use)
# ---------------------------------------------------------------------------

def _compute_statistics(
    fields_train: np.ndarray,   # (N_samples_total, C)  — all training points in RAM
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-channel normalization statistics over training data held in RAM.

    Returns (mean, std, robust_offset, robust_scale_99).
    NaN values are excluded from all computations.

    .. note::
        For large datasets use ``_WelfordAccumulator`` + ``_ReservoirSampler``
        instead; ``convert()`` already does this incrementally.
    """
    C = fields_train.shape[-1]
    mean   = np.empty(C, dtype=np.float32)
    std    = np.empty(C, dtype=np.float32)
    offset = np.empty(C, dtype=np.float32)
    scale  = np.empty(C, dtype=np.float32)

    for c in range(C):
        col = fields_train[:, c]
        col = col[np.isfinite(col)]
        if col.size == 0:
            mean[c] = std[c] = offset[c] = scale[c] = 0.0
            continue
        mean[c]   = float(np.mean(col))
        std[c]    = float(np.std(col))
        offset[c] = float(np.median(col))
        p1, p99   = np.percentile(col, [1.0, 99.0])
        scale[c]  = float(max(p99 - p1, 1e-8))

    return mean, std, offset, scale


# ---------------------------------------------------------------------------
# HDF5 writer — legacy / small-data path (kept for reference / testing)
# ---------------------------------------------------------------------------

def _write_hdf5(
    output_path: str,
    all_fields: np.ndarray,          # (B, T, N_ocean, C)  float32
    all_times: np.ndarray,           # (T,) float64  — shared time axis
    coordinates: np.ndarray,         # (N_ocean, 3)  float32
    var_names: List[str],
    level: int,
    subset_name: str,
    train_indices: List[int],
    val_indices: List[int],
    test_indices: List[int],
    bbox: Optional[Tuple[float, float, float, float]] = None,
    bbox_grid_shape: Optional[Tuple[int, int]] = None,
    ocean_grid_indices: Optional[np.ndarray] = None,
) -> None:
    """
    Write the full HDF5 dataset from an in-memory tensor (legacy path).

    .. deprecated::
        Use the incremental write path in ``convert()`` for large datasets
        to avoid the double-memory spike of ``np.stack`` + HDF5 write.
        This function is kept for small-dataset testing and reference.
    """
    import h5py

    B, T, N_ocean, C = all_fields.shape

    if train_indices:
        train_flat = all_fields[train_indices].reshape(-1, C)
        t_mean, t_std, t_offset, t_scale99 = _compute_statistics(train_flat)
    else:
        t_mean    = np.zeros(C, dtype=np.float32)
        t_std     = np.ones(C,  dtype=np.float32)
        t_offset  = np.zeros(C, dtype=np.float32)
        t_scale99 = np.ones(C,  dtype=np.float32)

    fields_6d = all_fields[:, :, :, np.newaxis, np.newaxis, :]
    coords_4d = coordinates[:, np.newaxis, np.newaxis, :]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tmp_path = output_path + ".partial"

    with h5py.File(tmp_path, "w") as f:
        f.create_dataset("fields", data=fields_6d,
                         chunks=(1, 1, N_ocean, 1, 1, C),
                         compression="gzip", compression_opts=4, shuffle=True)
        f.create_dataset("coordinates", data=coords_4d,
                         compression="gzip", compression_opts=4)
        f.create_dataset("time", data=all_times.astype(np.float64))
        cond = np.full((B, 1), float(level), dtype=np.float32)
        f.create_dataset("conditions", data=cond)

        tid_dt = h5py.special_dtype(vlen=str)
        tid_ds = f.create_dataset("trajectory_id", (B,), dtype=tid_dt)
        for b in range(B):
            tid_ds[b] = f"trajectory_{b:06d}"

        splits = f.require_group("splits")
        splits.create_dataset("train",      data=np.array(train_indices, dtype=np.int64))
        splits.create_dataset("validation", data=np.array(val_indices,   dtype=np.int64))
        splits.create_dataset("test",       data=np.array(test_indices,  dtype=np.int64))

        stats = f.require_group("statistics")
        stats.create_dataset("train_mean",       data=t_mean)
        stats.create_dataset("train_std",        data=t_std)
        stats.create_dataset("channel_offset",   data=t_offset)
        stats.create_dataset("channel_scale_99", data=t_scale99)

        f.attrs["field_names"] = json.dumps(list(var_names))
        f.attrs["field_units"] = json.dumps(["unknown"] * C)
        f.attrs["case_name"]   = subset_name
        f.attrs["grid_shape"]  = json.dumps([N_ocean])
        f.attrs["cb_vars"]     = json.dumps(list(var_names))
        f.attrs["cb_level"]    = int(level)
        if bbox is not None:
            f.attrs["cb_bbox"] = json.dumps(list(bbox))
        if bbox_grid_shape is not None:
            f.attrs["bbox_grid_shape"] = json.dumps(list(bbox_grid_shape))

        if ocean_grid_indices is not None:
            aux_grp = f.require_group("auxiliary")
            aux_grp.create_dataset(
                "ocean_grid_indices",
                data=ocean_grid_indices.astype(np.int32),
                compression="gzip", compression_opts=4,
            )

    os.replace(tmp_path, output_path)
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"  wrote {output_path}  ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Convert-info YAML
# ---------------------------------------------------------------------------

def _write_convert_info(
    output_dir: str,
    output_name: str,
    input_dir: str,
    var_names: List[str],
    level: int,
    dates: List[str],
    n_ocean: int,
    t_per_day: int,
    t_stored: int,
    time_stride: int,
    hdf5_path: str,
    split_fracs: Tuple[float, float, float],
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> None:
    """Write convert_info.yaml, mirroring carve_cb_heavy.py's subset_info.yaml."""
    import yaml
    from datetime import datetime as _dt

    info = {
        "output_name":    output_name,
        "input_dir":      os.path.abspath(input_dir),
        "output_dir":     os.path.abspath(output_dir),
        "hdf5_file":      os.path.abspath(hdf5_path),
        "vars":           list(var_names),
        "level":          int(level),
        "n_dates":        len(dates),
        "dates":          sorted(dates),
        "n_ocean_points": int(n_ocean),
        "t_per_day":      int(t_per_day),
        "t_stored":       int(t_stored),
        "time_stride":    int(time_stride),
        "split_fracs":    list(split_fracs),
        "bbox":           list(bbox) if bbox is not None else None,
        "created":        _dt.now().isoformat(timespec="seconds"),
        "yaml_snippet": {
            "data":         os.path.abspath(hdf5_path),
            "cb_var_names": list(var_names),
            "cb_var_level": int(level),
            "time_stride":  int(time_stride),
        },
    }
    out_path = os.path.join(output_dir, _INFO_FILENAME)
    with open(out_path, "w") as f:
        yaml.dump(info, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def _load_manifest(output_dir: str) -> dict:
    path = os.path.join(output_dir, _MANIFEST_FILENAME)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"converted": False, "dates": []}


def _save_manifest(output_dir: str, manifest: dict) -> None:
    path = os.path.join(output_dir, _MANIFEST_FILENAME)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def _make_splits(
    n_dates: int,
    split_fracs: Tuple[float, float, float],
) -> Tuple[List[int], List[int], List[int]]:
    """
    Assign trajectory indices to train / validation / test splits.

    Splits are chronological: first ``train_frac`` dates → train,
    next ``val_frac`` → validation, remainder → test.
    """
    train_frac, val_frac, _ = split_fracs
    train_end = max(1, int(n_dates * train_frac))
    val_end   = max(train_end, int(n_dates * (train_frac + val_frac)))
    train_idx = list(range(0, train_end))
    val_idx   = list(range(train_end, val_end))
    test_idx  = list(range(val_end, n_dates))
    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Main conversion routine — incremental (memory-safe) write
# ---------------------------------------------------------------------------

def convert(
    input_dir: str,
    output_dir: str,
    output_name: str,
    var_names: List[str],
    level: int,
    force: bool,
    split_fracs: Tuple[float, float, float],
    bbox: Optional[Tuple[float, float, float, float]] = None,
    grid_bbox: Optional[Tuple[int, int, int, int]] = None,
    always_wet: bool = False,
    time_stride: int = 1,
) -> int:
    """
    Convert a carved NC subset directory to HDF5 using incremental writes.

    Fields are written one day at a time — no large in-memory tensor is
    assembled.  The peak RAM footprint is approximately:

        max(T_per_day / time_stride, 1) × N_ocean × C × 4 bytes per date
        + O(C) for Welford accumulators
        + _RESERVOIR_SIZE × C × 4 bytes for the reservoir

    For N_ocean≈150k, T_stored≈48, C=4 this is ~115 MB per date plus ~32 MB
    for the reservoir, well within a 16 GB allocation.

    Parameters
    ----------
    bbox : (lon_min, lon_max, lat_min, lat_max) or None
        Geographic bounding box.  Mutually exclusive with grid_bbox.
    grid_bbox : (eta_min, eta_max, xi_min, xi_max) or None
        ROMS grid-index bounding box (0-based, inclusive).  Produces a perfect
        rectangle in (eta, xi) space so the training imshow shows a non-skewed
        domain.
    always_wet : bool
        When True, do a pre-pass reading wetdry_mask_rho and intersect with
        mask_rho so only points ALWAYS wet are retained.
    time_stride : int
        Store every ``time_stride``-th snapshot.  Default 1 (all snapshots).
        T_stored = len(range(0, T_per_day, time_stride)).

    Returns 0 on success, non-zero on failure.
    """
    import h5py
    import netCDF4 as nc  # noqa: F401  (validates import early)

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    manifest = _load_manifest(output_dir)

    hdf5_path = os.path.join(output_dir, _HDF5_FILENAME)
    if not force and manifest.get("converted") and os.path.exists(hdf5_path):
        print(f"  [skip] {hdf5_path} already exists — use --force to re-convert")
        return 0

    # ------------------------------------------------------------------
    # 1.  Discover carved daily files
    # ------------------------------------------------------------------
    carved_manifest_path = os.path.join(input_dir, "carved_manifest.json")
    if os.path.exists(carved_manifest_path):
        with open(carved_manifest_path) as f:
            carved_manifest = json.load(f)
        dates = sorted(
            d for d, e in carved_manifest.get("carved", {}).items() if e.get("done")
        )
    else:
        # Fallback: find YYYYMMDD.nc files by name pattern
        import re
        dates = sorted(
            p.stem for p in Path(input_dir).glob("????????.nc")
            if re.fullmatch(r"\d{8}", p.stem)
        )

    if not dates:
        print(f"Error: no carved daily files found in {input_dir}", file=sys.stderr)
        return 1

    print(f"  found {len(dates)} carved date(s)")

    # ------------------------------------------------------------------
    # 2.  Load static.nc → ocean mask
    # ------------------------------------------------------------------
    static_nc = os.path.join(input_dir, "static.nc")
    if not os.path.exists(static_nc):
        print(f"Error: static.nc not found in {input_dir}", file=sys.stderr)
        return 1

    # wetdry_mask_rho is only used for masking, never as a training channel.
    training_vars: List[str] = [v for v in var_names if v != "wetdry_mask_rho"]
    if len(training_vars) < len(var_names):
        print("  [info] 'wetdry_mask_rho' removed from training channels "
              "(used for always-wet masking only)")

    print("[static.nc]")
    ocean_mask, lon_full, lat_full, ocean_flat_idx, lon_ocean, lat_ocean = _load_static(static_nc)
    eta, xi = ocean_mask.shape

    # Apply geographic lon/lat bounding-box crop when requested
    if bbox is not None and grid_bbox is None:
        ocean_flat_idx, lon_ocean, lat_ocean = _apply_bbox_mask(
            ocean_mask, lon_full, lat_full, bbox
        )
        if ocean_flat_idx.size == 0:
            print("Error: bounding box contains zero ocean points.", file=sys.stderr)
            return 1

    # Apply grid-index bounding-box crop when requested
    if grid_bbox is not None:
        eta_min, eta_max, xi_min, xi_max = grid_bbox
        if eta_min < 0 or eta_max >= eta or eta_min > eta_max:
            print(f"Error: --grid-bbox eta range [{eta_min}..{eta_max}] is invalid "
                  f"(grid has {eta} rows, 0-based).", file=sys.stderr)
            return 1
        if xi_min < 0 or xi_max >= xi or xi_min > xi_max:
            print(f"Error: --grid-bbox xi range [{xi_min}..{xi_max}] is invalid "
                  f"(grid has {xi} columns, 0-based).", file=sys.stderr)
            return 1
        ocean_flat_idx, lon_ocean, lat_ocean = _apply_grid_bbox_mask(
            ocean_mask, lon_full, lat_full, eta_min, eta_max, xi_min, xi_max
        )
        if ocean_flat_idx.size == 0:
            print("Error: grid bbox contains zero ocean points.", file=sys.stderr)
            return 1

    # Always-wet pre-pass
    if always_wet:
        print("\n[always-wet pre-pass]")
        pre_mask = np.zeros(ocean_mask.shape, dtype=bool)
        pre_mask.ravel()[ocean_flat_idx] = True
        persistent_mask = _build_persistent_wet_mask(dates, input_dir, pre_mask)
        ocean_flat_idx = np.flatnonzero(persistent_mask).astype(np.int64)
        lon_ocean = lon_full.ravel()[ocean_flat_idx]
        lat_ocean = lat_full.ravel()[ocean_flat_idx]
        if ocean_flat_idx.size == 0:
            print("Error: always-wet filter removed all ocean points. "
                  "Check that wetdry_mask_rho is present in the carved files.",
                  file=sys.stderr)
            return 1

    N_ocean = ocean_flat_idx.shape[0]

    # Compute 2D grid indices for each ocean point (for visualization)
    ocean_row_idx = (ocean_flat_idx // xi).astype(np.int32)
    ocean_col_idx = (ocean_flat_idx % xi).astype(np.int32)
    row_min = int(ocean_row_idx.min()) if N_ocean > 0 else 0
    col_min = int(ocean_col_idx.min()) if N_ocean > 0 else 0
    ocean_grid_indices = np.stack(
        [ocean_row_idx - row_min, ocean_col_idx - col_min], axis=-1
    )  # (N_ocean, 2)  int32, local coords within bbox

    if grid_bbox is not None:
        eta_min_gb, eta_max_gb, xi_min_gb, xi_max_gb = grid_bbox
        bbox_ny = eta_max_gb - eta_min_gb + 1
        bbox_nx = xi_max_gb  - xi_min_gb  + 1
    else:
        bbox_ny = int(ocean_row_idx.max() - row_min + 1) if N_ocean > 0 else 0
        bbox_nx = int(ocean_col_idx.max() - col_min + 1) if N_ocean > 0 else 0

    bbox_grid_shape_hw: Optional[Tuple[int, int]] = (bbox_ny, bbox_nx)
    print(f"  bbox grid  : {bbox_ny} rows × {bbox_nx} cols  "
          f"(ocean fill {100.0 * N_ocean / max(bbox_ny * bbox_nx, 1):.1f} %)"
          + ("  [exact rectangle]" if grid_bbox is not None else ""))

    # Coordinates: (lon, lat, z=0) → (N_ocean, 3)
    z_ocean   = np.zeros(N_ocean, dtype=np.float32)
    coords    = np.stack([lon_ocean, lat_ocean, z_ocean], axis=-1)  # (N_ocean, 3)
    coords_4d = coords[:, np.newaxis, np.newaxis, :]                 # (N_ocean, 1, 1, 3)

    # ------------------------------------------------------------------
    # 3.  Probe T_per_day (reads only the time axis of one file)
    # ------------------------------------------------------------------
    print(f"\n[probe]")
    T_per_day = _probe_one_date(dates, input_dir)
    T_stored  = len(range(0, T_per_day, time_stride))
    print(f"  T_per_day  : {T_per_day}  ×  stride {time_stride}  →  T_stored = {T_stored}")

    # ------------------------------------------------------------------
    # 4.  Pre-scan: filter to dates with existing .nc files
    #     Used to pre-compute train/val/test split indices before the write
    #     loop — needed so we know which trajectories are "train" for online
    #     stats accumulation.
    # ------------------------------------------------------------------
    dates_existing = [d for d in dates
                      if os.path.exists(os.path.join(input_dir, f"{d}.nc"))]
    B_expected = len(dates_existing)
    if B_expected == 0:
        print("Error: no carved .nc files exist on disk.", file=sys.stderr)
        return 1

    train_idx_exp, val_idx_exp, test_idx_exp = _make_splits(B_expected, split_fracs)
    train_set = set(train_idx_exp)   # trajectory indices that belong to train
    print(f"  B_expected : {B_expected}  "
          f"(train≈{len(train_idx_exp)}, val≈{len(val_idx_exp)}, "
          f"test≈{len(test_idx_exp)})")

    # ------------------------------------------------------------------
    # 5.  Open HDF5 .partial file, pre-allocate fields dataset,
    #     then write one trajectory per date
    # ------------------------------------------------------------------
    C        = len(training_vars)
    # Use a per-process unique .partial name so that two concurrent jobs
    # targeting the same output never O_TRUNC each other's in-flight file.
    # The HDF5 file-lock (errno=11 / EAGAIN) protects against concurrent
    # writes, but on POSIX filesystems the O_TRUNC happens *before* the lock
    # is acquired, silently wiping the other job's data.
    tmp_path = hdf5_path + f".partial_{os.getpid()}"
    print(f"\n[HDF5 write → {hdf5_path}]")
    print(f"  pre-alloc  : ({B_expected}, {T_stored}, {N_ocean}, 1, 1, {C})  float32")

    # Online stats accumulators (training trajectories only)
    welford   = _WelfordAccumulator(C)
    reservoir = _ReservoirSampler(C, size=_RESERVOIR_SIZE)

    first_times: Optional[np.ndarray] = None
    good_dates:  List[str] = []

    with h5py.File(tmp_path, "w") as f:
        # Pre-allocate the fields dataset on disk — no data loaded into RAM.
        # maxshape allows resize if fewer dates succeed than expected.
        ds_fields = f.create_dataset(
            "fields",
            shape    =(B_expected, T_stored, N_ocean, 1, 1, C),
            maxshape =(B_expected, T_stored, N_ocean, 1, 1, C),
            dtype    =np.float32,
            chunks   =(1, 1, N_ocean, 1, 1, C),   # one spatial plane per chunk
            compression     ="gzip",
            compression_opts=4,
            shuffle  =True,
            fillvalue=np.nan,
        )

        actual_b = 0
        for date in dates_existing:
            nc_path = os.path.join(input_dir, f"{date}.nc")
            print(f"  [write] {date} …", end="", flush=True)
            try:
                t_vals, fields_day = _read_daily(
                    nc_path, training_vars, ocean_flat_idx, (eta, xi)
                )
            except Exception as exc:
                print(f" FAILED: {exc}", file=sys.stderr)
                continue

            # Pad / truncate to T_per_day for consistency
            if t_vals.shape[0] != T_per_day:
                if t_vals.shape[0] < T_per_day:
                    pad        = T_per_day - t_vals.shape[0]
                    t_vals     = np.concatenate([t_vals,
                                                 np.full(pad, np.nan)])
                    fields_day = np.concatenate(
                        [fields_day,
                         np.full((pad, N_ocean, C), np.nan, dtype=np.float32)])
                else:
                    t_vals     = t_vals[:T_per_day]
                    fields_day = fields_day[:T_per_day]

            # Apply time stride
            fields_strided = fields_day[::time_stride]   # (T_stored, N_ocean, C)
            t_strided      = t_vals[::time_stride]       # (T_stored,)

            if first_times is None:
                first_times = t_strided

            # Write directly to the pre-allocated HDF5 dataset on disk.
            # Shape written: (1, T_stored, N_ocean, 1, 1, C) at row actual_b.
            ds_fields[actual_b, :, :, 0, 0, :] = fields_strided

            good_dates.append(date)

            # Update online stats for training trajectories only
            if actual_b in train_set:
                welford.update(fields_strided)
                reservoir.update(fields_strided)

            actual_b += 1
            print(f" ok  (shape {fields_strided.shape})")

        # --------------------------------------------------------------
        # Resize if fewer dates succeeded than expected
        # --------------------------------------------------------------
        B_actual = actual_b
        if B_actual == 0:
            print("Error: no dates were read successfully.", file=sys.stderr)
            return 1

        if B_actual < B_expected:
            print(f"  [resize] {B_actual}/{B_expected} dates succeeded; "
                  f"resizing fields {B_expected}→{B_actual}.")
            ds_fields.resize((B_actual, T_stored, N_ocean, 1, 1, C))

        # Recompute splits for the actual number of trajectories
        train_idx, val_idx, test_idx = _make_splits(B_actual, split_fracs)

        # --------------------------------------------------------------
        # Finalize online statistics
        # --------------------------------------------------------------
        t_mean, t_std     = welford.finalize()
        t_offset, t_scale = reservoir.finalize()

        if not train_idx:
            # Edge case: no training trajectories (should not happen in practice)
            t_mean   = np.zeros(C, dtype=np.float32)
            t_std    = np.ones(C,  dtype=np.float32)
            t_offset = np.zeros(C, dtype=np.float32)
            t_scale  = np.ones(C,  dtype=np.float32)

        # --------------------------------------------------------------
        # Write all remaining (small) datasets
        # --------------------------------------------------------------
        f.create_dataset("coordinates", data=coords_4d,
                         compression="gzip", compression_opts=4)

        times_to_write = first_times if first_times is not None else np.zeros(T_stored)
        f.create_dataset("time", data=times_to_write.astype(np.float64))

        cond = np.full((B_actual, 1), float(level), dtype=np.float32)
        f.create_dataset("conditions", data=cond)

        tid_dt = h5py.special_dtype(vlen=str)
        tid_ds = f.create_dataset("trajectory_id", (B_actual,), dtype=tid_dt)
        for b in range(B_actual):
            tid_ds[b] = f"trajectory_{b:06d}"

        splits_grp = f.require_group("splits")
        splits_grp.create_dataset("train",
                                  data=np.array(train_idx, dtype=np.int64))
        splits_grp.create_dataset("validation",
                                  data=np.array(val_idx,   dtype=np.int64))
        splits_grp.create_dataset("test",
                                  data=np.array(test_idx,  dtype=np.int64))

        stats_grp = f.require_group("statistics")
        stats_grp.create_dataset("train_mean",       data=t_mean)
        stats_grp.create_dataset("train_std",        data=t_std)
        stats_grp.create_dataset("channel_offset",   data=t_offset)
        stats_grp.create_dataset("channel_scale_99", data=t_scale)

        # Attributes
        f.attrs["field_names"]         = json.dumps(list(training_vars))
        f.attrs["field_units"]         = json.dumps(["unknown"] * C)
        f.attrs["case_name"]           = output_name
        f.attrs["grid_shape"]          = json.dumps([N_ocean])
        f.attrs["cb_vars"]             = json.dumps(list(training_vars))
        f.attrs["cb_level"]            = int(level)
        f.attrs["time_stride_stored"]  = int(time_stride)
        if bbox is not None:
            f.attrs["cb_bbox"]         = json.dumps(list(bbox))
        if bbox_grid_shape_hw is not None:
            f.attrs["bbox_grid_shape"] = json.dumps(list(bbox_grid_shape_hw))

        # Auxiliary: 2D grid indices for visualization
        aux_grp = f.require_group("auxiliary")
        aux_grp.create_dataset(
            "ocean_grid_indices",
            data=ocean_grid_indices.astype(np.int32),
            compression="gzip", compression_opts=4,
        )

    # Atomically replace .partial with final filename
    os.replace(tmp_path, hdf5_path)
    size_mb = os.path.getsize(hdf5_path) / 1e6
    print(f"\n  wrote {hdf5_path}  ({size_mb:.1f} MB)")
    print(f"  tensor     : fields=({B_actual}, {T_stored}, {N_ocean}, 1, 1, {C})")
    print(f"  splits     : train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
    print(f"  stats mean : {t_mean}")
    print(f"  stats std  : {t_std}")

    # ------------------------------------------------------------------
    # 6.  Write convert_info.yaml
    # ------------------------------------------------------------------
    print(f"\n[{_INFO_FILENAME}]")
    _write_convert_info(
        output_dir  = output_dir,
        output_name = output_name,
        input_dir   = input_dir,
        var_names   = training_vars,
        level       = level,
        dates       = good_dates,
        n_ocean     = N_ocean,
        t_per_day   = T_per_day,
        t_stored    = T_stored,
        time_stride = time_stride,
        hdf5_path   = hdf5_path,
        split_fracs = split_fracs,
        bbox        = bbox,
    )

    # Update manifest
    manifest["converted"] = True
    manifest["dates"]     = sorted(good_dates)
    _save_manifest(output_dir, manifest)

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _auto_output_name(
    input_dir: str,
    with_bbox: bool = False,
    with_grid_bbox: bool = False,
) -> str:
    """Derive HDF5 output subdirectory name from the NC subset name."""
    name = Path(input_dir).resolve().name
    if name.endswith("_h5"):
        name = name[:-3]
    if with_grid_bbox:
        return name + "_grid_h5"
    if with_bbox:
        return name + "_bbox_h5"
    return name + "_h5"


def _read_subset_info(input_dir: str) -> Optional[dict]:
    """Try to load subset_info.yaml written by carve_cb_heavy.py."""
    try:
        import yaml
        path = os.path.join(input_dir, "subset_info.yaml")
        if os.path.exists(path):
            with open(path) as f:
                return yaml.safe_load(f)
    except Exception:
        pass
    return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert carved CoastalBench NetCDF subsets to HDF5.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Input / Output ---
    p.add_argument(
        "--input-dir", "-i", required=True,
        help=(
            "Directory produced by carve_cb_heavy.py, containing "
            "YYYYMMDD.nc files, static.nc, and carved_manifest.json."
        ),
    )
    p.add_argument(
        "--pool-dir", default=None,
        help=(
            "Base pool directory for HDF5 output. "
            "The HDF5 subdirectory is created inside this directory. "
            "Defaults to the parent of --input-dir."
        ),
    )
    p.add_argument(
        "--output-name", default=None, metavar="NAME",
        help=(
            "Explicit name for the HDF5 output subdirectory. "
            "Defaults to '<nc_subset_name>_h5' (auto-derived from --input-dir)."
        ),
    )

    # --- Variables and level (auto-detected from subset_info.yaml if omitted) ---
    p.add_argument(
        "--vars", nargs="+", default=None,
        choices=sorted(ALL_CB_VARS),
        metavar="VAR",
        help=(
            "Variables to include in the HDF5 file. "
            "Auto-detected from subset_info.yaml if omitted. "
            "Choices: " + " ".join(sorted(ALL_CB_VARS))
        ),
    )
    p.add_argument(
        "--level", type=int, default=None,
        help=(
            "Sigma level (for labelling). "
            "Auto-detected from subset_info.yaml if omitted."
        ),
    )

    # --- Temporal subsampling ---
    p.add_argument(
        "--time-stride", type=int, default=1, metavar="N",
        help=(
            "Store every Nth snapshot in the HDF5 (conversion-time subsampling). "
            "T_stored = len(range(0, T_per_day, N)).  "
            "For 48 snapshots/day (30-min output): "
            "  1 → 48 snaps (30-min, default); "
            "  2 → 24 snaps (1-hr); "
            "  4 → 12 snaps (2-hr); "
            "  6 →  8 snaps (3-hr). "
            "Independent of the training-time time_stride in dataset.yaml, "
            "which sub-selects from whatever is stored here at load time. "
            "Default: 1 (store all snapshots)."
        ),
    )

    # --- Spatial subregion ---
    p.add_argument(
        "--bbox", nargs=4, type=float, default=None,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        help=(
            "Geographic bounding box (decimal degrees) for spatial subregion crop. "
            "Applied on top of the ocean mask so only wet points inside the box "
            "are retained.  Example (Mid-Atlantic): --bbox -76 -70 35 42. "
            "CoastalBench uses negative values for west longitudes. "
            "Mutually exclusive with --grid-bbox."
        ),
    )
    p.add_argument(
        "--grid-bbox", nargs=4, type=int, default=None,
        metavar=("ETA_MIN", "ETA_MAX", "XI_MIN", "XI_MAX"),
        help=(
            "ROMS grid-index bounding box (0-based, inclusive).  Selects points "
            "by their (eta, xi) position in the curvilinear grid rather than by "
            "lon/lat.  The selected region is a perfect rectangle in (eta, xi) "
            "space so the training imshow shows a non-skewed, straight domain. "
            "Use visualize_cb_domain.py --grid-lines to read off indices. "
            "Mutually exclusive with --bbox."
        ),
    )

    # --- Splits ---
    p.add_argument(
        "--split-fracs", nargs=3, type=float, default=[0.8, 0.1, 0.1],
        metavar=("TRAIN", "VAL", "TEST"),
        help=(
            "Fraction of dates assigned to train / validation / test splits. "
            "Must sum to 1.0. Default: 0.8 0.1 0.1"
        ),
    )

    # --- Masking ---
    p.add_argument(
        "--always-wet", action="store_true",
        help=(
            "Only retain ocean points that are wet (wetdry_mask_rho == 1) in "
            "EVERY carved daily file.  Eliminates tidal flats and barrier islands "
            "that go dry during any timestep.  Requires 'wetdry_mask_rho' to be "
            "present in the carved NC files (include it in --vars when running "
            "carve_cb_heavy.py).  'wetdry_mask_rho' is never written as a training "
            "channel regardless of whether it appears in --vars."
        ),
    )

    # --- Control ---
    p.add_argument(
        "--force", action="store_true",
        help="Re-convert even if the output HDF5 already exists.",
    )

    return p


def main() -> int:
    args = build_parser().parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        print(f"Error: --input-dir '{input_dir}' does not exist.", file=sys.stderr)
        return 1

    # Validate time_stride
    if args.time_stride < 1:
        print("Error: --time-stride must be >= 1.", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Resolve vars and level (from CLI or subset_info.yaml)
    # ------------------------------------------------------------------
    subset_info = _read_subset_info(input_dir)

    var_names: List[str]
    if args.vars is not None:
        var_names = list(args.vars)
    elif subset_info is not None and "vars" in subset_info:
        var_names = list(subset_info["vars"])
        print(f"  auto-detected vars from subset_info.yaml: {var_names}")
    else:
        var_names = list(DEFAULT_VARS)
        print(f"  no subset_info.yaml found; using default vars: {var_names}")

    level: int
    if args.level is not None:
        level = args.level
    elif subset_info is not None and "level" in subset_info:
        level = int(subset_info["level"])
        print(f"  auto-detected level from subset_info.yaml: {level}")
    else:
        level = DEFAULT_LEVEL
        print(f"  no subset_info.yaml found; using default level: {level}")

    # ------------------------------------------------------------------
    # Resolve bbox / grid_bbox (mutually exclusive)
    # ------------------------------------------------------------------
    bbox: Optional[Tuple[float, float, float, float]] = None
    grid_bbox: Optional[Tuple[int, int, int, int]] = None

    if args.bbox is not None and args.grid_bbox is not None:
        print("Error: --bbox and --grid-bbox are mutually exclusive.", file=sys.stderr)
        return 1

    if args.bbox is not None:
        lon_min, lon_max, lat_min, lat_max = args.bbox
        if lon_min >= lon_max:
            print("Error: --bbox lon_min must be less than lon_max.", file=sys.stderr)
            return 1
        if lat_min >= lat_max:
            print("Error: --bbox lat_min must be less than lat_max.", file=sys.stderr)
            return 1
        bbox = (lon_min, lon_max, lat_min, lat_max)

    if args.grid_bbox is not None:
        eta_min, eta_max, xi_min, xi_max = args.grid_bbox
        if eta_min >= eta_max:
            print("Error: --grid-bbox ETA_MIN must be less than ETA_MAX.", file=sys.stderr)
            return 1
        if xi_min >= xi_max:
            print("Error: --grid-bbox XI_MIN must be less than XI_MAX.", file=sys.stderr)
            return 1
        grid_bbox = (eta_min, eta_max, xi_min, xi_max)

    # ------------------------------------------------------------------
    # Resolve output directory
    # ------------------------------------------------------------------
    output_name = args.output_name or _auto_output_name(
        input_dir,
        with_bbox=(bbox is not None),
        with_grid_bbox=(grid_bbox is not None),
    )
    pool_dir   = os.path.abspath(args.pool_dir) if args.pool_dir else str(Path(input_dir).parent)
    output_dir = os.path.join(pool_dir, output_name)

    # ------------------------------------------------------------------
    # Validate split fracs
    # ------------------------------------------------------------------
    split_fracs_raw = args.split_fracs
    total = sum(split_fracs_raw)
    if abs(total - 1.0) > 1e-6:
        print(
            f"Error: --split-fracs must sum to 1.0 (got {split_fracs_raw} = {total:.4f}).",
            file=sys.stderr,
        )
        return 1
    split_fracs = (split_fracs_raw[0], split_fracs_raw[1], split_fracs_raw[2])

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\nInput NC subset  : {input_dir}")
    print(f"Output HDF5 dir  : {output_dir}")
    print(f"HDF5 file        : {os.path.join(output_dir, _HDF5_FILENAME)}")
    print(f"Variables        : {var_names}")
    print(f"Sigma level      : {level}")
    print(f"Time stride      : {args.time_stride}  "
          f"(store every {args.time_stride}th snapshot at conversion time)")
    if grid_bbox is not None:
        print(f"Bbox crop        : grid-index  eta[{grid_bbox[0]}..{grid_bbox[1]}]  "
              f"xi[{grid_bbox[2]}..{grid_bbox[3]}]  "
              f"→ {grid_bbox[1]-grid_bbox[0]+1}×{grid_bbox[3]-grid_bbox[2]+1} rectangle")
    elif bbox is not None:
        print(f"Bbox crop        : lon [{bbox[0]:.2f}, {bbox[1]:.2f}]  "
              f"lat [{bbox[2]:.2f}, {bbox[3]:.2f}]")
    else:
        print(f"Bbox crop        : (none — full domain)")
    print(f"Always-wet mask  : "
          f"{'yes (pre-pass over all wetdry_mask_rho)' if args.always_wet else 'no (static mask_rho only)'}")
    print(f"Split fracs      : train={split_fracs[0]:.2f}  val={split_fracs[1]:.2f}  "
          f"test={split_fracs[2]:.2f}")
    print()

    # ------------------------------------------------------------------
    # Run conversion
    # ------------------------------------------------------------------
    rc = convert(
        input_dir   = input_dir,
        output_dir  = output_dir,
        output_name = output_name,
        var_names   = var_names,
        level       = level,
        force       = args.force,
        split_fracs = split_fracs,
        bbox        = bbox,
        grid_bbox   = grid_bbox,
        always_wet  = args.always_wet,
        time_stride = args.time_stride,
    )

    if rc == 0:
        print(f"\n{'='*60}")
        print(f"Conversion complete.")
        print(f"HDF5 file    : {os.path.join(output_dir, _HDF5_FILENAME)}")
        print(f"Manifest     : {os.path.join(output_dir, _MANIFEST_FILENAME)}")
        print(f"Info         : {os.path.join(output_dir, _INFO_FILENAME)}")
        print(
            f"\n{'─'*60}\n"
            f"To use this dataset, set in your training YAML:\n"
            f"{'─'*60}\n"
            f"data: \"{os.path.join(output_dir, _HDF5_FILENAME)}\"\n"
            f"{'─'*60}"
        )
    return rc


if __name__ == "__main__":
    sys.exit(main())
