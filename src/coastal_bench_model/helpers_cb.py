"""
CoastalBench dataset helpers for GL_RBF_enh reconstruction.

This module is the CoastalBench analogue of helpers.py from the TC demo.
It exposes the same public interface so that downstream training and
evaluation scripts can treat both datasets identically:

    dataset.coords          (N_ocean, 2)  — normalized lon/lat in [0,1]^2
    dataset.coords_raw      (N_ocean, 2)  — raw lon/lat in degrees
    dataset.num_points      int
    dataset.num_fields      int
    dataset.field_names     tuple[str, ...]
    dataset.mean            (C,)  float32 tensor
    dataset.std             (C,)  float32 tensor
    dataset[i]  →  dict with keys: coords, coords_raw, fields,
                                   time_index, day_idx, step_in_day

Public helpers:
    normalize_coords(coords)
    build_sparse_condition(...)          — identical copy from helpers.py
    build_sparse_condition_cb(...)       — thin CB wrapper (placeholder)
    collate_cb(batch)                   — collate_fn for DataLoader
"""

from __future__ import annotations

import csv
import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import xarray as xr
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Variable / dimension constants  (confirmed from inspect_nc.py output)
# ---------------------------------------------------------------------------

# 3-D fields: dims = (ocean_time, s, eta, xi)  shape per day: (48, 12, 898, 598)
CB_3D_VARS = ("u", "v", "w", "temp", "salt", "AKs")

# 2-D fields: dims = (ocean_time, eta, xi)  shape per day: (48, 898, 598)
# Note: met variables (Pair…Vwind) are stored as float64 in the NC file.
CB_2D_VARS = ("zeta", "Pair", "Qair", "Tair", "rain", "swrad", "lwrad", "Uwind", "Vwind")

# Compact subset recommended for initial experiments
DEFAULT_CB_VARS  = ("u", "v", "temp", "zeta")
DEFAULT_CB_NAMES = ("u_surf", "v_surf", "temp_surf", "zeta")

# ROMS sigma-coordinate convention: s=0 is the deepest level (bottom),
# s=11 is the shallowest level (surface) in a 12-level model output.
DEFAULT_CB_LEVEL = 11

# xarray dimension name for the time axis
NC_TIME_DIM = "ocean_time"   # 48 half-hourly steps per daily file

# ---------------------------------------------------------------------------
# Coordinate normalization (identical to helpers.py)
# ---------------------------------------------------------------------------

def normalize_coords(coords: torch.Tensor) -> torch.Tensor:
    """Min-max normalize coordinates to [0, 1]^D per axis."""
    cmin  = coords.min(dim=0).values
    cmax  = coords.max(dim=0).values
    scale = (cmax - cmin).clamp_min(1e-8)
    return (coords - cmin) / scale


# ---------------------------------------------------------------------------
# Sparse conditioning helpers (copied verbatim from helpers.py)
# ---------------------------------------------------------------------------

def _to_int_list(x: Union[int, Sequence[int], None]) -> list[int]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    return [int(x)]


def _broadcast_per_field(
    values: Union[int, Sequence[int]],
    cond_fields: Sequence[int],
    name: str,
) -> list[int]:
    values = _to_int_list(values)
    if len(values) == 1:
        values = values * len(cond_fields)
    if len(values) != len(cond_fields):
        raise ValueError(
            f"{name} must have length 1 or match len(cond_fields). "
            f"Got {len(values)} vs {len(cond_fields)}."
        )
    return values


def build_sparse_condition(
    coords_full: torch.Tensor,
    fields_full: torch.Tensor,
    cond_fields: Union[int, Sequence[int]],
    n_obs_min: Union[int, Sequence[int]],
    n_obs_max: Union[int, Sequence[int]],
):
    """
    Generalized sparse conditioning.

    Args:
        coords_full: [B, N, D]
        fields_full: [B, N, C]
        cond_fields: int or list[int], e.g. 2 or [0, 2]
        n_obs_min: int or list[int], per conditioned field
        n_obs_max: int or list[int], per conditioned field

    Returns:
        obs_coords:    [B, M, D]
        obs_values:    [B, M, 1]
        obs_mask:      [B, M]
        obs_indices:   [B, M]
        obs_field_ids: [B, M]   # which field each sensor belongs to
    """
    cond_fields = _to_int_list(cond_fields)
    if len(cond_fields) == 0:
        raise ValueError("cond_fields must contain at least one field index.")

    n_obs_min = _broadcast_per_field(n_obs_min, cond_fields, "n_obs_min")
    n_obs_max = _broadcast_per_field(n_obs_max, cond_fields, "n_obs_max")

    for a, b in zip(n_obs_min, n_obs_max):
        if b < a:
            raise ValueError(f"Each n_obs_max must be >= n_obs_min, got {a} and {b}.")

    bsz, n_pts, coord_dim = coords_full.shape
    device = coords_full.device

    max_obs = sum(n_obs_max)

    obs_coords = torch.zeros(
        bsz, max_obs, coord_dim, device=device, dtype=coords_full.dtype
    )
    obs_values = torch.zeros(
        bsz, max_obs, 1, device=device, dtype=fields_full.dtype
    )
    obs_mask = torch.zeros(
        bsz, max_obs, device=device, dtype=coords_full.dtype
    )
    obs_indices = torch.zeros(
        bsz, max_obs, device=device, dtype=torch.long
    )
    obs_field_ids = torch.full(
        (bsz, max_obs), -1, device=device, dtype=torch.long
    )

    for b in range(bsz):
        cursor = 0
        for fld, nmin, nmax in zip(cond_fields, n_obs_min, n_obs_max):
            m   = int(torch.randint(low=nmin, high=nmax + 1, size=(1,), device=device).item())
            idx = torch.randperm(n_pts, device=device)[:m].sort().values

            obs_coords[b, cursor:cursor + m]       = coords_full[b, idx]
            obs_values[b, cursor:cursor + m, 0]    = fields_full[b, idx, fld]
            obs_mask[b, cursor:cursor + m]         = 1.0
            obs_indices[b, cursor:cursor + m]      = idx
            obs_field_ids[b, cursor:cursor + m]    = fld

            cursor += m

    return obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids


def build_sparse_condition_cb(
    coords_full: torch.Tensor,
    fields_full: torch.Tensor,
    cond_fields: Union[int, Sequence[int]],
    n_obs_min: Union[int, Sequence[int]],
    n_obs_max: Union[int, Sequence[int]],
    valid_indices=None,   # reserved: if None, all N_ocean points are eligible
    strategy: str = "random",  # placeholder for future strategies
):
    """
    CB-specific wrapper around build_sparse_condition.

    Since __getitem__ already returns only ocean points, every point in the
    batch is a valid sensor location.  valid_indices and strategy are kept
    as forward-compatible arguments but are not yet used.
    """
    return build_sparse_condition(
        coords_full, fields_full, cond_fields, n_obs_min, n_obs_max
    )


# ---------------------------------------------------------------------------
# Collate function for DataLoader
# ---------------------------------------------------------------------------

def collate_cb(batch: list[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate a list of CoastalBenchNCDataset samples into a batched dict.

    Replaces the collate used in train_pointcloud_ffm.py, which stacked
    'physical_time'.  CoastalBench uses 'day_idx' and 'step_in_day' instead.
    """
    return {
        "coords":      torch.stack([b["coords"]      for b in batch]),
        "fields":      torch.stack([b["fields"]      for b in batch]),
        "time_index":  torch.stack([b["time_index"]  for b in batch]),
        "day_idx":     torch.stack([b["day_idx"]     for b in batch]),
        "step_in_day": torch.stack([b["step_in_day"] for b in batch]),
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CoastalBenchNCDataset(Dataset):
    """
    Point-cloud dataset over the CoastalBench ROMS ocean simulation.

    Each sample is one time-step (30 min) from one daily .nc file, restricted
    to ocean (wet) grid points as defined by static.nc mask_rho.

    Directory layout assumed::

        data_dir/
            20080101.nc
            20080102.nc
            ...
        static_nc_path  →  static.nc  (contains lon, lat, mask_rho)

    Parameters
    ----------
    data_dir:
        Directory containing daily YYYYMMDD.nc files.
    static_nc_path:
        Path to static.nc (provides lon, lat, mask_rho).
    split:
        "train", "val", or "test".
    train_ratio:
        Fraction of days assigned to the training split.  Splitting is done
        at the day level (not the timestep level) to prevent temporal leakage.
    seed:
        RNG seed for the day-level shuffle.
    var_names:
        Which NC variables to load as model fields.
    var_level:
        Vertical s-index for 3-D variables.  Defaults to DEFAULT_CB_LEVEL=11
        (surface in ROMS bottom-to-top convention).
    field_names:
        Human-readable labels for var_names, used by downstream metrics /
        visualizations.  Must be the same length as var_names.
    stats_path:
        Path for the cached per-field statistics file (.pt).  If None,
        defaults to <data_dir>/cb_dataset.stats.pt.
    stats_chunk_days:
        Number of day-files to process per chunk when computing stats.
    time_stride:
        Sub-sample every Nth time-step within each day (1 = all 48 steps).
    use_wetdry_mask:
        If True, additionally exclude points that are dry in ANY timestep
        of ANY training day (uses wetdry_mask_rho from daily files).
        Expensive on first call; the refined mask is stored in ocean_mask.
        If False, only the static mask_rho is used.
    """

    # Maximum number of open xarray datasets to keep in the file-handle cache.
    _CACHE_SIZE = 3

    def __init__(
        self,
        data_dir: str,
        static_nc_path: str,
        split: str = "train",
        train_ratio: float = 0.85,
        seed: int = 42,
        var_names: Tuple[str, ...] = DEFAULT_CB_VARS,
        var_level: int = DEFAULT_CB_LEVEL,
        field_names: Tuple[str, ...] = DEFAULT_CB_NAMES,
        stats_path: Optional[str] = None,
        stats_chunk_days: int = 5,
        time_stride: int = 1,
        use_wetdry_mask: bool = False,
    ) -> None:
        super().__init__()

        self.data_dir        = Path(data_dir)
        self.static_nc_path  = Path(static_nc_path)
        self.split           = split
        self.train_ratio     = float(train_ratio)
        self.seed            = int(seed)
        self.var_names       = tuple(var_names)
        self.var_level       = int(var_level)
        self.field_names     = tuple(field_names)
        self.stats_chunk_days = int(stats_chunk_days)
        self.time_stride     = int(time_stride)
        self.use_wetdry_mask = bool(use_wetdry_mask)
        self.num_fields      = len(self.var_names)

        # File handle cache: maps day_idx → open xr.Dataset
        self._ds_cache: dict[int, xr.Dataset] = {}
        self._cache_order: list[int] = []    # insertion order for eviction

        # ------------------------------------------------------------------
        # 1. Enumerate and sort daily NC files
        # Restrict to YYYYMMDD.nc (8-digit date stems) to exclude static.nc
        # and any other auxiliary files that may live in the same directory.
        # ------------------------------------------------------------------
        import re as _re
        _date_pat = _re.compile(r"^\d{8}\.nc$")
        self.day_files: List[Path] = sorted(
            p for p in self.data_dir.glob("*.nc") if _date_pat.match(p.name)
        )
        if len(self.day_files) == 0:
            raise FileNotFoundError(
                f"No YYYYMMDD.nc files found in data_dir: {self.data_dir}"
            )
        n_days = len(self.day_files)

        # ------------------------------------------------------------------
        # 2. Day-level train / val-test split  (shuffle by seed, split by day)
        # ------------------------------------------------------------------
        rng = np.random.default_rng(self.seed)
        all_day_indices = np.arange(n_days, dtype=np.int64)
        shuffled = rng.permutation(all_day_indices)
        n_train = int(n_days * self.train_ratio)

        if split == "train":
            split_day_indices = np.sort(shuffled[:n_train])
        elif split in ("val", "test"):
            split_day_indices = np.sort(shuffled[n_train:])
        else:
            raise ValueError(f"split must be 'train', 'val', or 'test'; got {split!r}")

        self._train_day_indices = np.sort(shuffled[:n_train])  # kept for stats

        # ------------------------------------------------------------------
        # 3. Load static.nc: lon, lat, mask_rho
        # ------------------------------------------------------------------
        with xr.open_dataset(self.static_nc_path, engine="netcdf4") as ds_static:
            # lon/lat are stored as xarray coordinates (confirmed by inspect_nc.py)
            lon_2d = ds_static.coords["lon"].values.astype(np.float32)   # (898, 598)
            lat_2d = ds_static.coords["lat"].values.astype(np.float32)   # (898, 598)
            # mask_rho is a data variable with values in {0., 1.}
            mask_2d = ds_static["mask_rho"].values.astype(np.float32)    # (898, 598)

        # Flatten spatial dims to (898*598,)
        lon_flat  = lon_2d.ravel()    # (N_total,)
        lat_flat  = lat_2d.ravel()
        mask_flat = mask_2d.ravel()

        # Boolean ocean mask (static)
        ocean_bool = mask_flat > 0.5

        # ------------------------------------------------------------------
        # 4. Optional: refine mask with wetdry_mask_rho across training days
        # ------------------------------------------------------------------
        if use_wetdry_mask:
            ocean_bool = self._refine_wetdry_mask(
                ocean_bool, self._train_day_indices
            )

        self.ocean_mask    = torch.from_numpy(ocean_bool)                  # (N_total,) bool
        self.ocean_indices = torch.where(self.ocean_mask)[0].long()        # (N_ocean,)
        self.num_points    = int(self.ocean_indices.shape[0])

        # ------------------------------------------------------------------
        # 5. Build normalized and raw coordinates for ocean points
        # ------------------------------------------------------------------
        lon_ocean = torch.from_numpy(lon_flat[ocean_bool])   # (N_ocean,)
        lat_ocean = torch.from_numpy(lat_flat[ocean_bool])   # (N_ocean,)

        coords_raw = torch.stack([lon_ocean, lat_ocean], dim=1)  # (N_ocean, 2)
        self.coords_raw = coords_raw
        self.coords     = normalize_coords(coords_raw)           # (N_ocean, 2) in [0,1]^2

        # ------------------------------------------------------------------
        # 6. Build sample_list: [(day_idx, step_idx), ...] in chrono order
        # ------------------------------------------------------------------
        self.sample_list: List[Tuple[int, int]] = []
        for day_idx in split_day_indices:          # already sorted chronologically
            for step_idx in range(0, 48, self.time_stride):
                self.sample_list.append((int(day_idx), int(step_idx)))

        # ------------------------------------------------------------------
        # 7. Per-field statistics (compute or load from cache)
        # ------------------------------------------------------------------
        default_stats = str(self.data_dir / "cb_dataset.stats.pt")
        self.stats_path = Path(stats_path) if stats_path is not None else Path(default_stats)
        self.mean, self.std = self._load_or_compute_stats()

    # ------------------------------------------------------------------
    # Wetdry mask refinement (optional)
    # ------------------------------------------------------------------

    def _refine_wetdry_mask(
        self, ocean_bool: np.ndarray, train_day_indices: np.ndarray
    ) -> np.ndarray:
        """
        Exclude any grid point that is ever dry across all training day files.

        A point is considered dry at a given timestep when wetdry_mask_rho == 0.
        We mark a point as permanently dry if it is dry in ANY training timestep.
        """
        ever_dry = np.zeros(ocean_bool.shape, dtype=bool)
        for day_idx in train_day_indices:
            path = self.day_files[day_idx]
            with xr.open_dataset(path, engine="netcdf4") as ds:
                wd = ds["wetdry_mask_rho"].values   # (48, 898, 598)
                # Dry = 0 in any step
                ever_dry_day = (wd == 0).any(axis=0).ravel()  # (N_total,)
                ever_dry |= ever_dry_day
        return ocean_bool & ~ever_dry

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _load_or_compute_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.stats_path.exists():
            obj = torch.load(self.stats_path, map_location="cpu", weights_only=True)
            return obj["mean"].float(), obj["std"].float()

        print(
            f"[CoastalBenchNCDataset] Computing per-field stats over "
            f"{len(self._train_day_indices)} training days …"
        )

        total_sum   = torch.zeros(self.num_fields, dtype=torch.float64)
        total_sq    = torch.zeros(self.num_fields, dtype=torch.float64)
        total_count = 0

        n_train_days = len(self._train_day_indices)
        for chunk_start in range(0, n_train_days, self.stats_chunk_days):
            chunk = self._train_day_indices[chunk_start : chunk_start + self.stats_chunk_days]
            for day_idx in chunk:
                path = self.day_files[int(day_idx)]
                with xr.open_dataset(path, engine="netcdf4") as ds:
                    for step_idx in range(0, 48, self.time_stride):
                        slice_vals = self._read_fields(ds, step_idx)  # (N_total, C)
                        # Apply ocean mask before accumulating
                        ocean_vals = slice_vals[self.ocean_indices.numpy()]  # (N_ocean, C)
                        x = torch.from_numpy(ocean_vals).double()
                        total_sum   += x.sum(dim=0)
                        total_sq    += (x ** 2).sum(dim=0)
                        total_count += x.shape[0]

        mean = (total_sum / total_count).float()
        var  = (total_sq / total_count - mean.double() ** 2).clamp_min(1e-12).float()
        std  = torch.sqrt(var)

        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mean": mean, "std": std}, self.stats_path)
        print(f"[CoastalBenchNCDataset] Stats saved to {self.stats_path}")
        return mean, std

    # ------------------------------------------------------------------
    # Internal field reader
    # ------------------------------------------------------------------

    def _read_fields(self, ds: xr.Dataset, step_idx: int) -> np.ndarray:
        """
        Read one time-step of all var_names from an open xr.Dataset.

        Returns an array of shape (N_total, C) = (898*598, len(var_names)),
        dtype float32.  Land points are NOT yet masked here; masking happens
        in the caller.  NaN / fill values are replaced with 0.0.

        Supports both full files (3-D vars have shape (ocean_time, s, eta, xi))
        and pre-carved slim files produced by carve_cb_heavy.py (3-D vars have
        shape (ocean_time, eta, xi), level already extracted at carve
        time).  Detection is automatic via the variable's ndim.
        """
        slices = []
        for var in self.var_names:
            if var in CB_3D_VARS:
                arr_var = ds[var]
                if arr_var.ndim == 4:
                    # Full file: (ocean_time, s, eta, xi)
                    arr = arr_var[step_idx, self.var_level, :, :].values  # (898, 598)
                else:
                    # Carved file: level already extracted → (ocean_time, eta, xi)
                    arr = arr_var[step_idx, :, :].values                  # (898, 598)
            else:
                arr = ds[var][step_idx, :, :].values                      # (898, 598)
            arr = arr.astype(np.float32)
            np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
            slices.append(arr.ravel())   # (898*598,)

        return np.stack(slices, axis=1)  # (N_total, C)

    # ------------------------------------------------------------------
    # NC file cache management
    # ------------------------------------------------------------------

    def _get_dataset(self, day_idx: int) -> xr.Dataset:
        """
        Return an open xr.Dataset for day_idx, using a small LRU-style cache.
        Evicts the oldest entry when the cache is full.
        """
        if day_idx in self._ds_cache:
            # Refresh recency
            self._cache_order.remove(day_idx)
            self._cache_order.append(day_idx)
            return self._ds_cache[day_idx]

        # Evict if at capacity
        if len(self._ds_cache) >= self._CACHE_SIZE:
            oldest = self._cache_order.pop(0)
            self._ds_cache[oldest].close()
            del self._ds_cache[oldest]

        path = self.day_files[day_idx]
        ds = xr.open_dataset(path, engine="netcdf4")
        self._ds_cache[day_idx] = ds
        self._cache_order.append(day_idx)
        return ds

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.sample_list)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        day_idx, step_idx = self.sample_list[i]

        ds = self._get_dataset(day_idx)
        raw = self._read_fields(ds, step_idx)             # (N_total, C)

        # Extract ocean points only
        ocean_np = raw[self.ocean_indices.numpy()]        # (N_ocean, C)

        # Normalize (z-score)
        x = torch.from_numpy(ocean_np)                   # (N_ocean, C) float32
        x = (x - self.mean) / self.std

        return {
            "coords":      self.coords.clone(),                             # (N_ocean, 2)
            "coords_raw":  self.coords_raw.clone(),                         # (N_ocean, 2)
            "fields":      x,                                               # (N_ocean, C)
            "time_index":  torch.tensor(i,        dtype=torch.long),
            "day_idx":     torch.tensor(day_idx,  dtype=torch.long),
            "step_in_day": torch.tensor(step_idx, dtype=torch.long),
        }

    def close(self) -> None:
        """Close all cached file handles.  Call when done with the dataset."""
        for ds in self._ds_cache.values():
            ds.close()
        self._ds_cache.clear()
        self._cache_order.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Obs-consistency comparison utilities (generic, no TC-specific code)
# Copied verbatim from 0_demo_TurbulentCombustion/src/helpers.py lines 601-743
# ---------------------------------------------------------------------------

def _save_figure_all_formats(fig, base_path: str, dpi: int = 250) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{base_path}.{ext}", dpi=dpi, bbox_inches="tight")


def save_obs_consistency_comparison(rows: list[dict], senconsis_dir: str) -> None:
    os.makedirs(senconsis_dir, exist_ok=True)
    csv_path  = os.path.join(senconsis_dir, "obs_consistency_comparison.csv")
    json_path = os.path.join(senconsis_dir, "obs_consistency_comparison.json")
    fieldnames = ["mode", "relative_l2", "obs_rel_l2_SenConsis", "obs_count_SenConsis_total"]
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    modes       = [row["mode"] for row in rows]
    sensor_vals = [row.get("obs_rel_l2_SenConsis", np.nan) for row in rows]
    full_vals   = [row.get("relative_l2", np.nan) for row in rows]
    x = np.arange(len(modes))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    width = 0.36
    ax.bar(x - width / 2, sensor_vals, width=width, label="obs_rel_l2_SenConsis")
    if np.any(np.isfinite(full_vals)):
        ax.bar(x + width / 2, full_vals, width=width, label="relative_l2")
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=20, ha="right")
    ax.set_ylabel("Relative L2")
    ax.set_title("Sensor consistency comparison")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_figure_all_formats(fig, os.path.join(senconsis_dir, "obs_consistency_comparison_bar"))
    plt.close(fig)
