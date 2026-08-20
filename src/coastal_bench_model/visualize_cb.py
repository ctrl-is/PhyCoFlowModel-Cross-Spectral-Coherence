"""
Grid-based visualization for CoastalBench reconstruction.

Replaces TC's Delaunay-triangulation / tricontourf approach with
np.unravel_index + imshow, which is both faster and exact for the ROMS
structured grid.

Public API
----------
visualize_reconstruction_cb        — single-field 3-panel imshow plot
visualize_reconstruction_cb_batch  — high-level drop-in for TC's
                                     visualize_reconstruction; handles
                                     model sampling + per-field dispatch
"""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

from helpers_cb import (
    build_sparse_condition,
    _to_int_list,
    _broadcast_per_field,
)

# Grid dimensions for CoastalBench ROMS output
_GRID_ETA = 898
_GRID_XI  = 598


# ---------------------------------------------------------------------------
# Low-level single-field plot
# ---------------------------------------------------------------------------

def visualize_reconstruction_cb(
    true_f: np.ndarray,                          # (N_ocean,) physical units
    pred_f: np.ndarray,                          # (N_ocean,) physical units
    ocean_indices: np.ndarray,                   # (N_ocean,) flat indices into (898, 598)
    coords_raw: np.ndarray,                      # (N_ocean, 2) lon/lat in degrees
    field_name: str,
    save_path: str,
    epoch: int = 0,
    sensor_coords_raw: Optional[np.ndarray] = None,  # (M, 2) lon/lat
    dpi: int = 200,
) -> float:
    """
    Save a 3-panel plot: Ground truth | Reconstruction | |Error|.

    Uses imshow on the (eta=898, xi=598) ROMS grid.  Land cells are
    pre-filled with np.nan and rendered as light grey.

    Returns
    -------
    float
        Normalized L2 error: ||truth - pred|| / (||truth|| + 1e-8).
    """
    # ------------------------------------------------------------------
    # Reconstruct 2-D grids
    # ------------------------------------------------------------------
    rows, cols = np.unravel_index(ocean_indices, (_GRID_ETA, _GRID_XI))

    grid_true = np.full((_GRID_ETA, _GRID_XI), np.nan, dtype=np.float32)
    grid_pred = np.full((_GRID_ETA, _GRID_XI), np.nan, dtype=np.float32)
    grid_err  = np.full((_GRID_ETA, _GRID_XI), np.nan, dtype=np.float32)

    grid_true[rows, cols] = true_f.astype(np.float32)
    grid_pred[rows, cols] = pred_f.astype(np.float32)
    grid_err[rows, cols]  = np.abs(true_f - pred_f).astype(np.float32)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    l2 = float(
        np.linalg.norm(true_f - pred_f) / (np.linalg.norm(true_f) + 1e-8)
    )

    # ------------------------------------------------------------------
    # Geographic extent from ocean point lon/lat
    # ------------------------------------------------------------------
    lon_min, lon_max = float(coords_raw[:, 0].min()), float(coords_raw[:, 0].max())
    lat_min, lat_max = float(coords_raw[:, 1].min()), float(coords_raw[:, 1].max())
    extent = [lon_min, lon_max, lat_min, lat_max]

    # ------------------------------------------------------------------
    # Colour limits
    # ------------------------------------------------------------------
    vmin = float(np.nanmin([grid_true.min(), grid_pred.min()]))
    vmax = float(np.nanmax([grid_true.max(), grid_pred.max()]))
    if vmin == vmax:
        vmin -= 1e-6
        vmax += 1e-6

    err_max = float(np.nanmax(grid_err)) if not np.all(np.isnan(grid_err)) else 1.0
    err_max = max(err_max, 1e-12)

    # ------------------------------------------------------------------
    # Colormaps with explicit NaN (land) colour
    # ------------------------------------------------------------------
    cmap_field = matplotlib.colormaps["RdBu_r"].copy()
    cmap_field.set_bad("lightgray")

    cmap_err = matplotlib.colormaps["hot_r"].copy()
    cmap_err.set_bad("lightgray")

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im_true = axes[0].imshow(
        grid_true, origin="upper", aspect="auto", extent=extent,
        cmap=cmap_field, vmin=vmin, vmax=vmax,
    )
    axes[0].set_title("Ground truth")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")

    if sensor_coords_raw is not None and len(sensor_coords_raw) > 0:
        axes[0].scatter(
            sensor_coords_raw[:, 0], sensor_coords_raw[:, 1],
            s=10, c="lime", edgecolors="black", linewidths=0.4,
            marker="o", zorder=4, label="sensors",
        )

    im_pred = axes[1].imshow(
        grid_pred, origin="upper", aspect="auto", extent=extent,
        cmap=cmap_field, vmin=vmin, vmax=vmax,
    )
    axes[1].set_title("Reconstruction")
    axes[1].set_xlabel("Longitude")

    im_err = axes[2].imshow(
        grid_err, origin="upper", aspect="auto", extent=extent,
        cmap=cmap_err, vmin=0.0, vmax=err_max,
    )
    axes[2].set_title("|Error|")
    axes[2].set_xlabel("Longitude")

    fig.colorbar(im_true, ax=axes[:2].tolist(), shrink=0.7, label=field_name)
    fig.colorbar(im_err,  ax=axes[2],           shrink=0.7, label=f"|{field_name} - û|")

    fig.suptitle(
        f"{field_name}   |   Normalized L2 = {l2:.3e}",
        fontsize=13,
    )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return l2


# ---------------------------------------------------------------------------
# High-level batch visualization (drop-in for TC's visualize_reconstruction)
# ---------------------------------------------------------------------------

@torch.no_grad()
def visualize_reconstruction_cb_batch(
    model: torch.nn.Module,
    dataset,                            # CoastalBenchNCDataset
    epoch: int,
    device: torch.device,
    save_dir: str,
    cond_fields: Union[int, Sequence[int]],
    n_obs: Union[int, Sequence[int]],
    n_steps: int = 32,
    ode_solver: Optional[str] = None,
    snapshot_index: int = 0,
    file_tag: Optional[str] = None,
    save_metrics_json: bool = True,
    return_payload: bool = False,
    **kwargs,                           # absorb obs_consistency_* kwargs from train_cb
) -> Union[Dict, tuple]:
    """
    Run model sampling on one validation snapshot and save per-field plots.

    This function is a drop-in replacement for TC's ``visualize_reconstruction``
    in the training loop.  The call signature is identical; only the
    internal visualization uses grid-based imshow instead of tricontourf.

    Parameters
    ----------
    model:
        Trained PointCloudFFM (or compatible) model.
    dataset:
        A CoastalBenchNCDataset instance; must expose ``ocean_indices``,
        ``coords_raw``, ``mean``, ``std``, and ``field_names``.
    epoch, device, save_dir, cond_fields, n_obs, n_steps, ode_solver,
    snapshot_index, file_tag, save_metrics_json, return_payload:
        Same semantics as TC's ``visualize_reconstruction``.
    **kwargs:
        Silently absorbed (obs_consistency_mode, obs_consistency_strength,
        obs_consistency_sigma, etc.) so the training-loop call site does
        not need to be edited.

    Returns
    -------
    dict
        Per-field normalized L2 errors, e.g. ``{"u_surf": 0.12, ...}``.
        If ``return_payload`` is True, returns ``(metrics, payload)``
        where ``payload`` is a dict of raw tensors / arrays.
    """
    model.eval()

    cond_fields_list = _to_int_list(cond_fields)
    n_obs_list       = _broadcast_per_field(n_obs, cond_fields_list, "n_obs")

    # ------------------------------------------------------------------
    # 1. Fetch one validation snapshot
    # ------------------------------------------------------------------
    sample      = dataset[snapshot_index]
    coords      = sample["coords"].unsqueeze(0).to(device)      # [1, N, 2]
    coords_raw  = sample["coords_raw"].unsqueeze(0)              # [1, N, 2] (CPU)
    truth       = sample["fields"].unsqueeze(0).to(device)       # [1, N, C]

    # ------------------------------------------------------------------
    # 2. Build sparse observations
    # ------------------------------------------------------------------
    obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition(
        coords_full = coords,
        fields_full = truth,
        cond_fields = cond_fields_list,
        n_obs_min   = n_obs_list,
        n_obs_max   = n_obs_list,   # exact count for visualization
    )

    # ------------------------------------------------------------------
    # 3. Model forward (sample)
    # ------------------------------------------------------------------
    sig = inspect.signature(model.sample)
    sample_kwargs: Dict = dict(
        coords      = coords,
        obs_coords  = obs_coords,
        obs_values  = obs_values,
        obs_mask    = obs_mask,
        n_steps     = n_steps,
        clamp_indices = obs_indices,
    )
    if "obs_field_ids" in sig.parameters:
        sample_kwargs["obs_field_ids"] = obs_field_ids
    if "ode_solver" in sig.parameters and ode_solver is not None:
        sample_kwargs["ode_solver"] = ode_solver

    # Pass through any obs_consistency kwargs that the model.sample accepts
    for key, val in kwargs.items():
        if key in sig.parameters:
            sample_kwargs[key] = val

    recon = model.sample(**sample_kwargs)   # [1, N, C] — normalized

    # ------------------------------------------------------------------
    # 4. Denormalize to physical units
    # ------------------------------------------------------------------
    mean = dataset.mean.to(device)   # (C,)
    std  = dataset.std.to(device)    # (C,)

    recon_phys = (recon * std.view(1, 1, -1) + mean.view(1, 1, -1))[0].cpu().numpy()  # [N, C]
    truth_phys = (truth * std.view(1, 1, -1) + mean.view(1, 1, -1))[0].cpu().numpy()  # [N, C]

    # ------------------------------------------------------------------
    # 5. Extract sensor positions in lon/lat for overlay
    # ------------------------------------------------------------------
    coords_raw_np = coords_raw[0].numpy()        # (N, 2)
    ocean_indices = dataset.ocean_indices.numpy()  # (N_ocean,) flat grid indices

    valid_mask = obs_mask[0].bool().cpu()
    obs_idx_valid  = obs_indices[0, valid_mask].cpu().numpy()
    obs_fid_valid  = obs_field_ids[0, valid_mask].cpu().numpy()

    # ------------------------------------------------------------------
    # 6. Per-field plots
    # ------------------------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)
    field_names = tuple(getattr(dataset, "field_names", [f"field_{c}" for c in range(truth_phys.shape[-1])]))
    metrics: Dict[str, float] = {}

    prefix = file_tag if file_tag is not None else f"epoch_{epoch:04d}"

    for c, fname in enumerate(field_names):
        sensor_coords_raw_c = None
        field_sensor_mask   = obs_fid_valid == c
        if np.any(field_sensor_mask):
            sensor_coords_raw_c = coords_raw_np[obs_idx_valid[field_sensor_mask]]  # (M, 2)

        save_path = os.path.join(save_dir, f"{prefix}_field_{fname}.png")

        l2 = visualize_reconstruction_cb(
            true_f          = truth_phys[:, c],
            pred_f          = recon_phys[:, c],
            ocean_indices   = ocean_indices,
            coords_raw      = coords_raw_np,
            field_name      = fname,
            save_path       = save_path,
            epoch           = epoch,
            sensor_coords_raw = sensor_coords_raw_c,
        )
        metrics[fname] = l2

    # ------------------------------------------------------------------
    # 7. Optional metrics JSON
    # ------------------------------------------------------------------
    if save_metrics_json:
        json_path = os.path.join(save_dir, f"{prefix}_metrics.json")
        payload_json = {
            "epoch":          int(epoch),
            "snapshot_index": int(snapshot_index),
            "cond_fields":    [int(v) for v in cond_fields_list],
            "n_obs":          [int(v) for v in n_obs_list],
            "n_steps":        int(n_steps),
            "ode_solver":     ode_solver,
            "metrics":        metrics,
        }
        with open(json_path, "w") as f:
            json.dump(payload_json, f, indent=2)

    if return_payload:
        payload = {
            "coords":         coords.cpu(),
            "coords_raw":     coords_raw,
            "truth":          truth.cpu(),
            "recon":          recon.cpu(),
            "truth_phys":     truth_phys,
            "recon_phys":     recon_phys,
            "obs_coords":     obs_coords.cpu(),
            "obs_values":     obs_values.cpu(),
            "obs_mask":       obs_mask.cpu(),
            "obs_indices":    obs_indices.cpu(),
            "obs_field_ids":  obs_field_ids.cpu(),
            "field_names":    list(field_names),
            "ocean_indices":  ocean_indices,
        }
        return metrics, payload

    return metrics
