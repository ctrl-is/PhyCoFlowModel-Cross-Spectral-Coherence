#!/usr/bin/env python
"""
Cross-spectral coherence for turbulent combustion

Examples
--------
  # model mode (GPU): reconstruct an ensemble from a checkpoint
  python cross_spectral_coherence.py --checkpoint <run_dir> --split val \
      --ensemble-size 24 --n-modes 384 --extra-pairs CO,T CH4,CO --save-stack stack

  # offline mode (no GPU): re-render any pair from a saved stack
  python cross_spectral_coherence.py --npz-stack "stack/*.npz" --extra-pairs T,p
"""
from __future__ import annotations

# Functions needed for Graph Basis and Coherence defintion
# Works when running either:
#   python cross_spectral/visualization.py
# or, depending on package setup:
#   python -m cross_spectral.visualization
try:
    from .graph import make_graph_frequency_bands
    from .cross_spectral import (
        CrossSpectralConfig,
        gft,
        inverse_graph_fourier_transform,
        compute_physical_coherence_loss,
    )
except ImportError:
    from graph import make_graph_frequency_bands
    from cross_spectral import (
        CrossSpectralConfig,
        gft,
        inverse_graph_fourier_transform,
        compute_physical_coherence_loss,
    )

import argparse
import glob
import inspect
import json
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# src/ on path so paper_style (figures) and evaluate_coherence (model mode) import.
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
for _d in (str(SCRIPT_DIR), str(SRC_DIR)):
    if _d not in sys.path:
        sys.path.append(_d)

# paper_style is optional. If the file is missing, use a safe fallback.
try:
    import paper_style as ps
except ModuleNotFoundError:
    class _PaperStyleFallback:
        CMAP_ERROR = "RdBu_r"

        @staticmethod
        def set_paper_style():
            plt.rcParams.update({
                "font.size": 8,
                "axes.titlesize": 9,
                "axes.labelsize": 8,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "legend.fontsize": 7,
                "figure.dpi": 150,
                "savefig.dpi": 400,
                "axes.linewidth": 0.7,
            })

    ps = _PaperStyleFallback()

EPS = 1e-12

# Repo bands are low/mid/high graph frequency.
# Low graph frequency = large spatial scale.
BAND_KEYS = ("low", "mid", "high")
BAND_DISPLAY = {
    "low": "large",
    "mid": "medium",
    "high": "small",
}

# ===========================================================================
# Spectral Basis
# ===========================================================================
@dataclass
class VizGraphBasis:
    eigvecs: np.ndarray
    eigvals: np.ndarray
    freqs: np.ndarray
    bands: Dict[str, np.ndarray]
    keep_axes: np.ndarray
    band_edges: np.ndarray
    band_names: Tuple[str, ...] = ("low", "mid", "high")

    @property
    def n_modes(self) -> int:
        return self.eigvecs.shape[1]

    def band_mode_mask(self, band: int | str) -> np.ndarray:
        """
        Compatibility helper for plotting code.

        Allows:
            basis.band_mode_mask(0)      -> low
            basis.band_mode_mask("low") -> low
        """
        if isinstance(band, int):
            band_name = self.band_names[band]
        else:
            band_name = band

        mask = np.zeros(self.n_modes, dtype=bool)
        mask[np.asarray(self.bands[band_name], dtype=int)] = True
        return mask


def effective_coords(
        coords: np.ndarray,
        var_tol: float = 1e-9,
        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Drop coordinate axes with near-zero variance for graph construction/plotting.
    """
    coords = np.asarray(coords, dtype=np.float64)
    var = coords.var(axis=0)
    keep_axes = np.where(var > var_tol)[0]

    if keep_axes.size == 0:
        keep_axes = np.arange(min(2, coords.shape[1]))

    return coords[:, keep_axes], keep_axes


def _band_edges_from_bands(
        freqs: np.ndarray,
        bands: Dict[str, np.ndarray],
        band_names: Tuple[str, ...] = ("low", "mid", "high"),
        ) -> np.ndarray:
    """
    Approximate frequency edges for shading plots.

    This is only for visualization. The real band assignment comes from graph.py.
    """
    edges = []

    for name in band_names:
        idx = np.asarray(bands[name], dtype=int)
        if idx.size == 0:
            continue
        edges.append(float(freqs[idx].min()))

    last_idx = np.asarray(bands[band_names[-1]], dtype=int)
    if last_idx.size > 0:
        edges.append(float(freqs[last_idx].max()))

    edges = np.asarray(edges, dtype=np.float64)

    if len(edges) != 4:
        # Fallback: equal visual thirds over non-DC frequencies.
        valid = np.arange(1, len(freqs))
        edges = np.linspace(float(freqs[valid].min()), float(freqs[valid].max()), 4)

    edges[0] -= EPS
    edges[-1] += EPS
    return edges


def effective_plot_axes(coords: np.ndarray, var_tol: float = 1e-9) -> np.ndarray:
    """
    Choose two non-degenerate coordinate axes for plotting only.
    """
    coords = np.asarray(coords, dtype=np.float64)
    var = coords.var(axis=0)
    keep_axes = np.where(var > var_tol)[0]

    if keep_axes.size >= 2:
        return keep_axes[:2]

    if keep_axes.size == 1:
        extra = [i for i in range(coords.shape[1]) if i != keep_axes[0]]
        return np.asarray([keep_axes[0], extra[0]], dtype=int)

    return np.arange(min(2, coords.shape[1]))


def load_viz_graph_basis(
        graph_basis_path: str | Path,
        current_coords: np.ndarray | None = None,
        ) -> VizGraphBasis:
    """
    Load precomputed graph basis from build_graph_basis.py output.
    """
    z = torch.load(graph_basis_path, map_location="cpu")

    eigvals = z["eigenvalues"].detach().cpu().numpy()
    U = z["U"].detach().cpu().numpy()
    basis_coords = z["coords"].detach().cpu().numpy()

    if current_coords is not None:
        current_coords = np.asarray(current_coords, dtype=np.float64)

        if current_coords.shape[0] != U.shape[0]:
            raise ValueError(
                f"Graph basis has {U.shape[0]} nodes, but current coords have "
                f"{current_coords.shape[0]} nodes. The basis and reconstructions "
                "do not match."
            )

        if current_coords.shape == basis_coords.shape:
            max_diff = float(np.max(np.abs(current_coords - basis_coords)))
            if max_diff > 1e-5:
                print(
                    "[csc] warning: current coords differ from saved basis coords; "
                    f"max |diff| = {max_diff:.3e}"
                )

        keep_axes = effective_plot_axes(current_coords)
    else:
        keep_axes = effective_plot_axes(basis_coords)

    bands = make_graph_frequency_bands(
        eigvals,
        exclude_zero=True,
        split="thirds",
    )

    freqs = np.sqrt(np.clip(eigvals, 0.0, None))
    band_names = tuple(bands.keys())
    band_edges = _band_edges_from_bands(freqs, bands, band_names)

    print(f"[csc] loaded graph basis: {graph_basis_path}")
    print(f"[csc] U shape: {U.shape}")
    print(f"[csc] eigenvalues shape: {eigvals.shape}")
    print(
        "[csc] modes per band: "
        + ", ".join(f"{BAND_DISPLAY[k]}={len(bands[k])}" for k in BAND_KEYS)
    )

    return VizGraphBasis(
        eigvecs=U,
        eigvals=eigvals,
        freqs=freqs,
        bands=bands,
        keep_axes=keep_axes,
        band_edges=band_edges,
        band_names=band_names,
    )

# ===========================================================================
# Coherence Metrics
# ===========================================================================
def field_pairs(n_fields: int) -> List[Tuple[int, int]]:
    return list(combinations(range(n_fields), 2))

def stack_to_tensor(stack: Sequence[np.ndarray], device: str = "cpu") -> torch.Tensor:
    return torch.as_tensor(np.stack(stack, axis=0), dtype=torch.float32, device=device)

def tensor_to_numpy_dict(d: Dict[str, object]) -> Dict[str, object]:
    out = {}
    for k, v in d.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu().numpy()
        elif isinstance(v, dict):
            out[k] = {
                kk: vv.detach().cpu().numpy() if torch.is_tensor(vv) else vv
                for kk, vv in v.items()
            }
        else:
            out[k] = v
    return out

def compute_repo_outputs(
        basis: VizGraphBasis,
        gt_stack: Sequence[np.ndarray],
        ffm_stack: Sequence[np.ndarray],
        eta_crossfreq: float = 1.0,
        device: str = "cpu",
        ) -> Dict[str, object]:
    U = torch.as_tensor(basis.eigvecs, dtype=torch.float32, device=device)
    gt = stack_to_tensor(gt_stack, device=device)
    ffm = stack_to_tensor(ffm_stack, device=device)

    cfg = CrossSpectralConfig(eta_crossfreq=eta_crossfreq)

    out = compute_physical_coherence_loss(
        fields_pred=ffm,
        fields_target=gt,
        U=U,
        bands=basis.bands,
        cfg=cfg,
    )
    return tensor_to_numpy_dict(out)

def band_limited_field(basis: VizGraphBasis, fields: np.ndarray, band: int | str) -> np.ndarray:
    """
    Band-pass reconstruction for spatial panel A using imported inverse GFT.
    """
    fields_t = torch.as_tensor(fields[None, ...], dtype=torch.float32)
    U_t = torch.as_tensor(basis.eigvecs, dtype=torch.float32)

    coeffs = gft(fields_t, U_t)  # [1, K, C]
    mask = basis.band_mode_mask(band)

    filt = torch.zeros_like(coeffs)
    filt[:, torch.as_tensor(mask), :] = coeffs[:, torch.as_tensor(mask), :]

    rec = inverse_graph_fourier_transform(filt, U_t)  # [1, N, C]
    return rec[0].detach().cpu().numpy()

def band_pair_values_from_coherence(
        coh: np.ndarray,
        basis: VizGraphBasis,
        pairs: List[Tuple[int, int]],
        ) -> np.ndarray:
    """
    Return band-averaged coherence as (3, n_pairs).
    """
    out = np.zeros((len(BAND_KEYS), len(pairs)), dtype=np.float64)

    for b, key in enumerate(BAND_KEYS):
        idx = np.asarray(basis.bands[key], dtype=int)
        for p, (i, j) in enumerate(pairs):
            out[b, p] = float(np.mean(coh[idx, i, j]))

    return out

def coherence_matrix(gamma2_band: np.ndarray, pairs: List[Tuple[int, int]], n_fields: int) -> np.ndarray:
    n_bands = gamma2_band.shape[0]
    mat = np.zeros((n_bands, n_fields, n_fields), dtype=np.float64)

    for b in range(n_bands):
        for d in range(n_fields):
            mat[b, d, d] = 1.0
        for p, (i, j) in enumerate(pairs):
            mat[b, i, j] = mat[b, j, i] = gamma2_band[b, p]

    return mat

def frequency_bins(basis: VizGraphBasis, n_bins: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    f = basis.freqs
    nz = np.arange(1, basis.n_modes)
    edges = np.linspace(f[nz].min(), f[nz].max(), n_bins + 1)
    edges[0] -= EPS

    bin_assignment = np.full(basis.n_modes, -1, dtype=np.int64)
    bin_assignment[nz] = np.clip(np.digitize(f[nz], edges[1:-1]), 0, n_bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return bin_assignment, centers

def binned_pair_coherence(
        coh: np.ndarray,
        basis: VizGraphBasis,
        pairs: List[Tuple[int, int]],
        n_bins: int,
        ) -> Tuple[np.ndarray, np.ndarray]:
    """Average imported coherence tensor into frequency bins for panel C curves."""
    bin_assignment, centers = frequency_bins(basis, n_bins=n_bins)
    n_pairs = len(pairs)
    out = np.zeros((len(centers), n_pairs), dtype=np.float64)

    for b in range(len(centers)):
        idx = np.where(bin_assignment == b)[0]
        if idx.size == 0:
            continue
        for p, (i, j) in enumerate(pairs):
            out[b, p] = float(np.mean(coh[idx, i, j]))

    return out, centers

def samefreq_loss_breakdown_from_coherence(
        coh_pred: np.ndarray,
        coh_target: np.ndarray,
        basis: VizGraphBasis,
        pairs: List[Tuple[int, int]],
        n_freq_bins: int,
        ) -> Dict[str, np.ndarray]:
    g_pred, centers = binned_pair_coherence(coh_pred, basis, pairs, n_freq_bins)
    g_target, _ = binned_pair_coherence(coh_target, basis, pairs, n_freq_bins)

    diff2 = (g_pred - g_target) ** 2

    bin_band = np.clip(np.digitize(centers, basis.band_edges[1:-1]), 0, 2)

    per_pair_band = np.zeros((3, len(pairs)), dtype=np.float64)
    for b in range(3):
        per_pair_band[b] = diff2[bin_band == b].sum(axis=0)

    per_pair = diff2.sum(axis=0)
    per_band = per_pair_band.sum(axis=1)
    total = float(per_pair.sum())

    return {
        "per_pair_band": per_pair_band,
        "per_pair": per_pair,
        "per_band": per_band,
        "total": total,
        "band_fraction": per_band / (total + 1e-9),
    }

def crossfreq_loss_breakdown_from_Q(
        Q_pred: np.ndarray,
        Q_target: np.ndarray,
        pairs: List[Tuple[int, int]],
        ) -> Dict[str, np.ndarray]:
    M, _, C, _ = Q_pred.shape
    off_diag = ~np.eye(M, dtype=bool)

    per_pair_bandpair = np.zeros((len(pairs), M, M), dtype=np.float64)
    per_pair = np.zeros(len(pairs), dtype=np.float64)

    for p, (i, j) in enumerate(pairs):
        diff2 = (Q_pred[:, :, i, j] - Q_target[:, :, i, j]) ** 2
        diff2 = diff2 * off_diag
        per_pair_bandpair[p] = diff2
        per_pair[p] = float(diff2[off_diag].mean())

    return {
        "per_pair_bandpair": per_pair_bandpair,
        "per_pair": per_pair,
        "total": float(per_pair.mean()) if len(per_pair) else 0.0,
        "off_diag_mask": off_diag,
    }

def rank_pairs_by_band_coherence(gb_gt: np.ndarray) -> np.ndarray:
    return np.argsort(-gb_gt.mean(axis=0))

# ===========================================================================
# Figures
# ===========================================================================
# Band background tints (match the channel-flow spectrum figure).
BAND_TINT = ("#e8e8f7", "#e7f2e7", "#f8e7e7")        # large / medium / small
BAND_EDGE_C = ("#5b5bd6", "#2e8b57", "#c0392b")
CMAP_FLUCT = "RdBu_r"                                  # zero-mean band field
CMAP_COFLUCT = "coolwarm"                              # f1*f2 (red = aligned)
CMAP_COH = "magma"                                     # coherence in [0, 1]
COL_GT = "#000000"
COL_FFM = "#D55E00"


@dataclass
class VizConfig:
    marker_size: float = 2.2
    n_freq_bins: int = 24
    clip_pct: float = 99.0
    dpi: int = 400
    point_alpha: float = 1.0


def pair_label(names: Sequence[str], i: int, j: int) -> str:
    return f"{names[i]}–{names[j]}"


# ---------------------------------------------------------------------------
# Low-level scatter
# ---------------------------------------------------------------------------
def _scatter(ax, xy, val, cmap, vmin, vmax, s):
    sc = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=val,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=s,
        marker="o",
        linewidths=0.0,
        rasterized=True,
    )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
    return sc


def _sym(vals, pct):
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return -1.0, 1.0
    a = float(np.percentile(np.abs(finite), pct)) or 1.0
    return -a, a


# ---------------------------------------------------------------------------
# Block A : spatial co-fluctuation
# ---------------------------------------------------------------------------
def draw_spatial_block(
        sub,
        xy: np.ndarray,
        basis: VizGraphBasis,
        fields_gt: np.ndarray,
        fields_ffm: np.ndarray,
        pair: Tuple[int, int],
        names: Sequence[str],
        gamma_gt: np.ndarray,       # (3,) band coherence for this pair, GT
        gamma_ffm: np.ndarray,      # (3,) band coherence for this pair, FFM
        cfg: VizConfig,
        title: Optional[str] = None,
        ) -> None:
    i, j = pair
    s = cfg.marker_size
    band_names = tuple(BAND_DISPLAY[k].capitalize() for k in BAND_KEYS)

    col_titles = [
        f"{names[i]}  (truth)",
        f"{names[j]}  (truth)",
        f"{names[i]}·{names[j]}  (truth)",
        f"{names[i]}·{names[j]}  (FFM)",
    ]

    gs = sub.add_gridspec(
        3,
        4,
        hspace=0.12,
        wspace=0.05,
        left=0.07,
        right=0.92,
        top=0.88,
        bottom=0.06,
    )

    for b in range(3):
        f_gt = band_limited_field(basis, fields_gt, b)       # (N, C)
        f_ff = band_limited_field(basis, fields_ffm, b)

        f1, f2 = f_gt[:, i], f_gt[:, j]

        def _cof(arr_i, arr_j):
            ni = arr_i / (arr_i.std() + 1e-12)
            nj = arr_j / (arr_j.std() + 1e-12)
            return ni * nj

        cof_gt = _cof(f_gt[:, i], f_gt[:, j])
        cof_ff = _cof(f_ff[:, i], f_ff[:, j])

        v1 = _sym(f1, cfg.clip_pct)
        v2 = _sym(f2, cfg.clip_pct)
        vcof = _sym(np.concatenate([cof_gt, cof_ff]), cfg.clip_pct)

        ax0 = sub.add_subplot(gs[b, 0])
        ax1 = sub.add_subplot(gs[b, 1])
        ax2 = sub.add_subplot(gs[b, 2])
        ax3 = sub.add_subplot(gs[b, 3])

        _scatter(ax0, xy, f1, CMAP_FLUCT, *v1, s)
        _scatter(ax1, xy, f2, CMAP_FLUCT, *v2, s)
        _scatter(ax2, xy, cof_gt, CMAP_COFLUCT, *vcof, s)
        _scatter(ax3, xy, cof_ff, CMAP_COFLUCT, *vcof, s)

        ax0.set_ylabel(
            band_names[b],
            fontsize=9.5,
            fontweight="bold",
            rotation=90,
            labelpad=6,
        )

        ax2.text(
            0.5,
            -0.04,
            f"$\\gamma^2$={gamma_gt[b]:.2f}",
            color=COL_GT,
            transform=ax2.transAxes,
            ha="center",
            va="top",
            fontsize=8.0,
        )
        ax3.text(
            0.5,
            -0.04,
            f"$\\gamma^2$={gamma_ffm[b]:.2f}",
            color=COL_FFM,
            transform=ax3.transAxes,
            ha="center",
            va="top",
            fontsize=8.0,
            fontweight="bold",
        )

        if b == 0:
            for ax, t in zip((ax0, ax1, ax2, ax3), col_titles):
                ax.set_title(t, fontsize=8.5, pad=4)

    cax1 = sub.add_axes([0.935, 0.50, 0.012, 0.34])
    cb1 = sub.figure.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(-1, 1), cmap=CMAP_FLUCT),
        cax=cax1,
    )
    cb1.set_ticks([-1, 0, 1])
    cb1.set_ticklabels(["–", "0", "+"])
    cb1.set_label("band fluctuation\n(per-panel scaled)", fontsize=7.0)
    cb1.ax.tick_params(labelsize=7.0)

    cax2 = sub.add_axes([0.935, 0.08, 0.012, 0.34])
    cb2 = sub.figure.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(-1, 1), cmap=CMAP_COFLUCT),
        cax=cax2,
    )
    cb2.set_ticks([-1, 0, 1])
    cb2.set_ticklabels(["oppose", "0", "align"])
    cb2.set_label("co-fluctuation", fontsize=7.0)
    cb2.ax.tick_params(labelsize=7.0)

    if title:
        sub.suptitle(title, fontsize=10.5, fontweight="bold", y=0.98)


# ---------------------------------------------------------------------------
# Block B : all-pair same-frequency scale matrices
# ---------------------------------------------------------------------------
def draw_matrix_block(
        sub,
        mat_gt: np.ndarray,         # (3, C, C) ground-truth coherence
        mat_err: np.ndarray,        # (3, C, C) FFM - GT
        names: Sequence[str],
        title: Optional[str] = None,
        ) -> None:
    band_names = tuple(BAND_DISPLAY[k] for k in BAND_KEYS)
    C = mat_gt.shape[1]
    err_lim = float(np.percentile(np.abs(mat_err), 99)) or 1.0

    gs = sub.add_gridspec(
        2,
        3,
        hspace=0.38,
        wspace=0.18,
        left=0.10,
        right=0.90,
        top=0.86,
        bottom=0.10,
    )

    def _draw_one(ax, M, kind, vmin, vmax, show_y):
        cmap = CMAP_COH if kind == "coh" else ps.CMAP_ERROR
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xticks(range(C))
        ax.set_yticks(range(C))
        ax.set_xticklabels(names, fontsize=7.0, rotation=45, ha="right")
        ax.set_yticklabels(names if show_y else [], fontsize=7.0)

        for a in range(C):
            for b in range(C):
                v = M[a, b]
                if kind == "coh":
                    txt = f"{v:.2f}"
                    light = v < 0.55
                else:
                    txt = f"{v:+.2f}"
                    light = abs(v) > 0.55 * vmax
                ax.text(
                    b,
                    a,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=5.6,
                    color="white" if light else "black",
                )
        return im

    im_gt = None
    for b in range(3):
        ax = sub.add_subplot(gs[0, b])
        im_gt = _draw_one(ax, mat_gt[b], "coh", 0.0, 1.0, show_y=(b == 0))
        ax.set_title(f"GT — {band_names[b]}", fontsize=8.5)

    im_err = None
    for b in range(3):
        ax = sub.add_subplot(gs[1, b])
        im_err = _draw_one(
            ax,
            mat_err[b],
            "err",
            -err_lim,
            err_lim,
            show_y=(b == 0),
        )
        ax.set_title(f"FFM–GT — {band_names[b]}", fontsize=8.5)

    cax1 = sub.add_axes([0.915, 0.52, 0.013, 0.30])
    cb1 = sub.figure.colorbar(im_gt, cax=cax1)
    cb1.set_label("$\\gamma^2$ coherence", fontsize=7.5)
    cb1.ax.tick_params(labelsize=7.0)

    cax2 = sub.add_axes([0.915, 0.12, 0.013, 0.30])
    cb2 = sub.figure.colorbar(im_err, cax=cax2)
    cb2.set_label("$\\Delta\\gamma^2$ (FFM–GT)", fontsize=7.5)
    cb2.ax.tick_params(labelsize=7.0)

    if title:
        sub.suptitle(title, fontsize=10.5, fontweight="bold", y=0.97)


# ---------------------------------------------------------------------------
# Block C : same-frequency coherence curves + per-pair loss bars
# ---------------------------------------------------------------------------
def _shade_bands(ax, band_edges):
    for b in range(3):
        ax.axvspan(
            band_edges[b],
            band_edges[b + 1],
            color=BAND_TINT[b],
            alpha=0.7,
            lw=0,
        )


def draw_curve_block(
        sub,
        basis: VizGraphBasis,
        coh_gt: np.ndarray,
        coh_ffm: np.ndarray,
        gb_gt: np.ndarray,
        gb_ffm: np.ndarray,
        pairs: List[Tuple[int, int]],
        pair_idx: int,
        pair: Tuple[int, int],
        names: Sequence[str],
        loss: Dict[str, np.ndarray],
        cfg: VizConfig,
        title: Optional[str] = None,
        ) -> None:
    g_gt, centers = binned_pair_coherence(coh_gt, basis, pairs, cfg.n_freq_bins)
    g_ff, _ = binned_pair_coherence(coh_ffm, basis, pairs, cfg.n_freq_bins)

    band_centers = [
        0.5 * (basis.band_edges[b] + basis.band_edges[b + 1])
        for b in range(3)
    ]

    gs = sub.add_gridspec(
        1,
        2,
        wspace=0.28,
        width_ratios=[1.05, 1.25],
        left=0.08,
        right=0.97,
        top=0.84,
        bottom=0.18,
    )

    axc = sub.add_subplot(gs[0, 0])
    _shade_bands(axc, basis.band_edges)

    axc.plot(
        centers,
        g_gt[:, pair_idx],
        color=COL_GT,
        lw=1.6,
        label="Ground truth",
    )
    axc.plot(
        centers,
        g_ff[:, pair_idx],
        color=COL_FFM,
        lw=1.6,
        ls="--",
        label="FFM",
    )

    axc.scatter(
        band_centers,
        gb_gt[:, pair_idx],
        color=COL_GT,
        s=34,
        zorder=5,
        edgecolors="white",
        linewidths=0.6,
    )
    axc.scatter(
        band_centers,
        gb_ffm[:, pair_idx],
        color=COL_FFM,
        s=34,
        zorder=5,
        marker="D",
        edgecolors="white",
        linewidths=0.6,
    )

    axc.set_xlabel("graph frequency  $\\nu_k=\\sqrt{\\lambda_k}$", fontsize=8.5)
    axc.set_ylabel("$\\gamma^2$ coherence", fontsize=8.5)
    axc.set_ylim(0, 1.02)
    axc.set_xlim(centers.min(), centers.max())
    axc.set_title(f"Coherence spectrum: {pair_label(names, *pair)}", fontsize=9)
    axc.legend(fontsize=7.5, loc="upper right", framealpha=0.85)

    for b, key in enumerate(BAND_KEYS):
        axc.text(
            band_centers[b],
            0.04,
            BAND_DISPLAY[key],
            ha="center",
            fontsize=7.0,
            color=BAND_EDGE_C[b],
            style="italic",
        )

    axb = sub.add_subplot(gs[0, 1])
    ppb = loss["per_pair_band"]
    plabels = [pair_label(names, a, b) for (a, b) in pairs]
    order = np.argsort(-loss["per_pair"])

    x = np.arange(len(plabels))
    bottom = np.zeros(len(plabels))

    for b, key in enumerate(BAND_KEYS):
        axb.bar(
            x,
            ppb[b][order],
            bottom=bottom,
            width=0.74,
            color=BAND_TINT[b],
            edgecolor=BAND_EDGE_C[b],
            lw=0.6,
            label=BAND_DISPLAY[key],
        )
        bottom += ppb[b][order]

    axb.set_xticks(x)
    axb.set_xticklabels(
        [plabels[k] for k in order],
        rotation=55,
        ha="right",
        fontsize=6.6,
    )
    axb.set_ylabel("$L_{\\mathrm{same}}^{c_1 c_2}$  (FFM vs GT)", fontsize=8.5)
    axb.set_title("Same-frequency loss decomposition by field pair", fontsize=9)
    axb.legend(
        fontsize=7.0,
        title="scale band",
        title_fontsize=7.0,
        loc="upper right",
    )
    axb.margins(x=0.01)

    if title:
        sub.suptitle(title, fontsize=10.5, fontweight="bold", y=0.99)


# ---------------------------------------------------------------------------
# Block D : cross-frequency / cross-band coupling heatmaps
# ---------------------------------------------------------------------------
def draw_crossfreq_block(
        sub,
        Q_gt: np.ndarray,
        Q_ffm: np.ndarray,
        pairs: List[Tuple[int, int]],
        pair: Tuple[int, int],
        names: Sequence[str],
        loss_xf: Dict[str, np.ndarray],
        title: Optional[str] = None,
        ) -> None:
    i, j = pair
    pair_idx = pairs.index(pair)

    M_gt = Q_gt[:, :, i, j]
    M_ff = Q_ffm[:, :, i, j]
    M_err = M_ff - M_gt

    band_labels = [BAND_DISPLAY[k].capitalize() for k in BAND_KEYS]
    err_lim = float(np.percentile(np.abs(M_err), 99)) or 1.0

    gs = sub.add_gridspec(
        1,
        4,
        width_ratios=[1.0, 1.0, 1.0, 0.9],
        wspace=0.35,
        left=0.08,
        right=0.95,
        top=0.78,
        bottom=0.18,
    )

    def _heat(ax, M, cmap, vmin, vmax, ttl, signed=False):
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xticks(range(len(band_labels)))
        ax.set_yticks(range(len(band_labels)))
        ax.set_xticklabels([f"{names[j]}\n{b}" for b in band_labels], fontsize=7.0)
        ax.set_yticklabels([f"{names[i]} {b}" for b in band_labels], fontsize=7.0)
        ax.set_title(ttl, fontsize=8.5)

        for r in range(M.shape[0]):
            for c in range(M.shape[1]):
                val = M[r, c]
                txt = f"{val:+.2f}" if signed else f"{val:.2f}"
                light = abs(val) > 0.55 * max(abs(vmin), abs(vmax), 1e-12)
                ax.text(
                    c,
                    r,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=7.0,
                    color="white" if light else "black",
                )

        # Diagonal band-pairs are same-band; off-diagonal is cross-frequency.
        for d in range(M.shape[0]):
            ax.add_patch(
                plt.Rectangle(
                    (d - 0.5, d - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="white",
                    linewidth=1.2,
                    linestyle="--",
                )
            )
        return im

    ax0 = sub.add_subplot(gs[0, 0])
    im0 = _heat(ax0, M_gt, CMAP_COH, 0.0, 1.0, "GT cross-band $Q$")

    ax1 = sub.add_subplot(gs[0, 1])
    _heat(ax1, M_ff, CMAP_COH, 0.0, 1.0, "FFM cross-band $Q$")

    ax2 = sub.add_subplot(gs[0, 2])
    im2 = _heat(
        ax2,
        M_err,
        ps.CMAP_ERROR,
        -err_lim,
        err_lim,
        "FFM–GT $\\Delta Q$",
        signed=True,
    )

    ax3 = sub.add_subplot(gs[0, 3])
    pp = loss_xf["per_pair_bandpair"][pair_idx].copy()
    off = loss_xf["off_diag_mask"]
    pp[~off] = np.nan

    im3 = ax3.imshow(pp, cmap="magma", aspect="equal")
    ax3.set_xticks(range(len(band_labels)))
    ax3.set_yticks(range(len(band_labels)))
    ax3.set_xticklabels(band_labels, fontsize=7.0)
    ax3.set_yticklabels(band_labels, fontsize=7.0)
    ax3.set_title("$L_{crossfreq}$ by band-pair", fontsize=8.5)

    for r in range(pp.shape[0]):
        for c in range(pp.shape[1]):
            if np.isfinite(pp[r, c]):
                ax3.text(
                    c,
                    r,
                    f"{pp[r, c]:.1e}",
                    ha="center",
                    va="center",
                    fontsize=5.8,
                    color="white",
                )

    cax1 = sub.add_axes([0.955, 0.49, 0.012, 0.25])
    cb1 = sub.figure.colorbar(im0, cax=cax1)
    cb1.set_label("$Q$", fontsize=7.0)
    cb1.ax.tick_params(labelsize=7.0)

    cax2 = sub.add_axes([0.955, 0.20, 0.012, 0.25])
    cb2 = sub.figure.colorbar(im2, cax=cax2)
    cb2.set_label("$\\Delta Q$", fontsize=7.0)
    cb2.ax.tick_params(labelsize=7.0)

    if title:
        sub.suptitle(title, fontsize=10.5, fontweight="bold", y=0.96)

def _top_cross_band_entries(
        Q_gt: np.ndarray,
        Q_ffm: np.ndarray,
        pair: Tuple[int, int],
        top_k: int = 4,
        mode: str = "strongest_gt",
        ) -> List[Tuple[int, int]]:
    """
    Select off-diagonal band-pairs for intuitive cross-frequency visualization.

    mode="strongest_gt": show strongest GT cross-band couplings.
    mode="largest_error": show cross-band couplings where FFM differs most.
    """
    i, j = pair
    M = Q_gt.shape[0]
    entries = []

    for a in range(M):
        for b in range(M):
            if a == b:
                continue

            if mode == "largest_error":
                score = abs(Q_ffm[a, b, i, j] - Q_gt[a, b, i, j])
            else:
                score = abs(Q_gt[a, b, i, j])

            entries.append((score, a, b))

    entries = sorted(entries, reverse=True)
    return [(a, b) for _, a, b in entries[:top_k]]


def draw_crossfreq_spatial_block(
        sub,
        xy: np.ndarray,
        basis: VizGraphBasis,
        fields_gt: np.ndarray,
        fields_ffm: np.ndarray,
        Q_gt: np.ndarray,
        Q_ffm: np.ndarray,
        pair: Tuple[int, int],
        names: Sequence[str],
        cfg: VizConfig,
        title: Optional[str] = None,
        top_k: int = 4,
        mode: str = "strongest_gt",
        ) -> None:
    """
    Intuitive spatial visualization of cross-frequency coupling.

    Each row shows one off-diagonal band interaction:
        field_i at band a  ×  field_j at band b

    Example:
        CH4-large with T-medium
        CH4-medium with T-large

    This directly visualizes the cross-frequency/cross-scale term rather than
    only showing the Q matrix.
    """
    i, j = pair
    s = cfg.marker_size

    entries = _top_cross_band_entries(
        Q_gt=Q_gt,
        Q_ffm=Q_ffm,
        pair=pair,
        top_k=top_k,
        mode=mode,
    )

    col_titles = [
        f"{names[i]} band $a$ (truth)",
        f"{names[j]} band $b$ (truth)",
        f"{names[i]}$_a$·{names[j]}$_b$ (truth)",
        f"{names[i]}$_a$·{names[j]}$_b$ (FFM)",
    ]

    gs = sub.add_gridspec(
        len(entries),
        4,
        hspace=0.20,
        wspace=0.05,
        left=0.07,
        right=0.92,
        top=0.88,
        bottom=0.08,
    )

    def _cof(arr_i, arr_j):
        ai = arr_i - arr_i.mean()
        aj = arr_j - arr_j.mean()
        ni = ai / (ai.std() + 1e-12)
        nj = aj / (aj.std() + 1e-12)
        return ni * nj

    for r, (a, b) in enumerate(entries):
        key_a = BAND_KEYS[a]
        key_b = BAND_KEYS[b]
        name_a = BAND_DISPLAY[key_a].capitalize()
        name_b = BAND_DISPLAY[key_b].capitalize()

        # GT cross-band fields
        f_gt_a = band_limited_field(basis, fields_gt, key_a)
        f_gt_b = band_limited_field(basis, fields_gt, key_b)

        # FFM cross-band fields
        f_ff_a = band_limited_field(basis, fields_ffm, key_a)
        f_ff_b = band_limited_field(basis, fields_ffm, key_b)

        # Cross-band product/co-fluctuation:
        # field i at band a with field j at band b.
        x_gt = f_gt_a[:, i]
        y_gt = f_gt_b[:, j]
        x_ff = f_ff_a[:, i]
        y_ff = f_ff_b[:, j]

        cof_gt = _cof(x_gt, y_gt)
        cof_ff = _cof(x_ff, y_ff)

        v1 = _sym(x_gt, cfg.clip_pct)
        v2 = _sym(y_gt, cfg.clip_pct)
        vcof = _sym(np.concatenate([cof_gt, cof_ff]), cfg.clip_pct)

        ax0 = sub.add_subplot(gs[r, 0])
        ax1 = sub.add_subplot(gs[r, 1])
        ax2 = sub.add_subplot(gs[r, 2])
        ax3 = sub.add_subplot(gs[r, 3])

        _scatter(ax0, xy, x_gt, CMAP_FLUCT, *v1, s)
        _scatter(ax1, xy, y_gt, CMAP_FLUCT, *v2, s)
        _scatter(ax2, xy, cof_gt, CMAP_COFLUCT, *vcof, s)
        _scatter(ax3, xy, cof_ff, CMAP_COFLUCT, *vcof, s)

        ax0.set_ylabel(
            f"{name_a} $\\to$\n{name_b}",
            fontsize=8.5,
            fontweight="bold",
            rotation=90,
            labelpad=8,
        )

        q_gt = Q_gt[a, b, i, j]
        q_ff = Q_ffm[a, b, i, j]
        dq = q_ff - q_gt

        ax2.text(
            0.5,
            -0.04,
            f"$Q_{{GT}}$={q_gt:.2f}",
            color=COL_GT,
            transform=ax2.transAxes,
            ha="center",
            va="top",
            fontsize=8.0,
        )
        ax3.text(
            0.5,
            -0.04,
            f"$Q_{{FFM}}$={q_ff:.2f}, $\\Delta$={dq:+.2f}",
            color=COL_FFM,
            transform=ax3.transAxes,
            ha="center",
            va="top",
            fontsize=8.0,
            fontweight="bold",
        )

        if r == 0:
            titles = [
                f"{names[i]} {name_a} (truth)",
                f"{names[j]} {name_b} (truth)",
                f"{names[i]} {name_a} · {names[j]} {name_b} (truth)",
                f"{names[i]} {name_a} · {names[j]} {name_b} (FFM)",
            ]
            for ax, t in zip((ax0, ax1, ax2, ax3), titles):
                ax.set_title(t, fontsize=8.5, pad=4)

    cax1 = sub.add_axes([0.935, 0.52, 0.012, 0.30])
    cb1 = sub.figure.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(-1, 1), cmap=CMAP_FLUCT),
        cax=cax1,
    )
    cb1.set_ticks([-1, 0, 1])
    cb1.set_ticklabels(["–", "0", "+"])
    cb1.set_label("band fluctuation\n(per-panel scaled)", fontsize=7.0)
    cb1.ax.tick_params(labelsize=7.0)

    cax2 = sub.add_axes([0.935, 0.14, 0.012, 0.30])
    cb2 = sub.figure.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(-1, 1), cmap=CMAP_COFLUCT),
        cax=cax2,
    )
    cb2.set_ticks([-1, 0, 1])
    cb2.set_ticklabels(["oppose", "0", "align"])
    cb2.set_label("cross-scale\nco-fluctuation", fontsize=7.0)
    cb2.ax.tick_params(labelsize=7.0)

    if title:
        sub.suptitle(title, fontsize=10.5, fontweight="bold", y=0.98)

def make_crossfreq_spatial_panels(
    out_dir: str | Path,
    xy: np.ndarray,
    basis: VizGraphBasis,
    fields_gt_repr: np.ndarray,
    fields_ffm_repr: np.ndarray,
    Q_gt: np.ndarray,
    Q_ffm: np.ndarray,
    names: Sequence[str],
    pairs: Sequence[Tuple[int, int]],
    cfg: Optional[VizConfig] = None,
    prefix: str = "cross_spectral_coherence",
    stacked: bool = True,
    top_k_bandpairs: int = 4,
    mode: str = "strongest_gt",
) -> Dict[str, Path]:
    """
    Save cross-frequency spatial panels for many field pairs.

    - one standalone figure per field pair
    - optionally one vertically stacked multipair figure
    """
    cfg = cfg or VizConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ps.set_paper_style()

    pairs = [tuple(sorted(p)) for p in pairs]
    saved: Dict[str, Path] = {}

    # One file per pair
    for pair in pairs:
        fig = plt.figure(figsize=(13.0, 5.8))
        draw_crossfreq_spatial_block(
            fig.subfigures(1, 1),
            xy,
            basis,
            fields_gt_repr,
            fields_ffm_repr,
            Q_gt,
            Q_ffm,
            pair,
            names,
            cfg,
            title=f"Intuitive cross-frequency spatial coupling: {pair_label(names, *pair)}",
            top_k=top_k_bandpairs,
            mode=mode,
        )

        p = out_dir / f"{prefix}_pair_{names[pair[0]]}_{names[pair[1]]}_crossfreq_spatial.png"
        fig.savefig(p, dpi=cfg.dpi, bbox_inches="tight")
        fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)

        key = f"crossfreq_pair_{names[pair[0]]}_{names[pair[1]]}"
        saved[key] = p
        saved[key + "_pdf"] = p.with_suffix(".pdf")

    # Optional stacked figure containing all requested pairs
    if stacked and len(pairs) > 1:
        fig = plt.figure(figsize=(13.0, 5.4 * len(pairs)))
        subs = fig.subfigures(len(pairs), 1)

        for sub, pair in zip(np.atleast_1d(subs), pairs):
            draw_crossfreq_spatial_block(
                sub,
                xy,
                basis,
                fields_gt_repr,
                fields_ffm_repr,
                Q_gt,
                Q_ffm,
                pair,
                names,
                cfg,
                title=f"Intuitive cross-frequency spatial coupling: {pair_label(names, *pair)}",
                top_k=top_k_bandpairs,
                mode=mode,
            )

        p = out_dir / f"{prefix}_multipair_crossfreq_spatial.png"
        fig.savefig(p, dpi=cfg.dpi, bbox_inches="tight")
        fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)

        saved["multipair_crossfreq_spatial_png"] = p
        saved["multipair_crossfreq_spatial_pdf"] = p.with_suffix(".pdf")

    return saved

# ---------------------------------------------------------------------------
# Optional standalone: all-pairs coherence small multiples
# ---------------------------------------------------------------------------
def figure_all_pair_curves(
        basis: VizGraphBasis,
        coh_gt: np.ndarray,
        coh_ffm: np.ndarray,
        pairs: List[Tuple[int, int]],
        names: Sequence[str],
        cfg: VizConfig,
        ) -> plt.Figure:
    g_gt, centers = binned_pair_coherence(coh_gt, basis, pairs, cfg.n_freq_bins)
    g_ff, _ = binned_pair_coherence(coh_ffm, basis, pairs, cfg.n_freq_bins)

    ncol = 5
    nrow = int(np.ceil(len(pairs) / ncol))
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(2.1 * ncol, 1.7 * nrow),
        squeeze=False,
    )

    for p, (i, j) in enumerate(pairs):
        ax = axes[p // ncol][p % ncol]
        _shade_bands(ax, basis.band_edges)
        ax.plot(centers, g_gt[:, p], color=COL_GT, lw=1.2)
        ax.plot(centers, g_ff[:, p], color=COL_FFM, lw=1.2, ls="--")
        ax.set_title(pair_label(names, i, j), fontsize=8)
        ax.set_ylim(0, 1.02)
        ax.set_xlim(centers.min(), centers.max())
        ax.tick_params(labelsize=6.5)
        if p % ncol == 0:
            ax.set_ylabel("$\\gamma^2$", fontsize=8)

    for q in range(len(pairs), nrow * ncol):
        axes[q // ncol][q % ncol].axis("off")

    fig.suptitle(
        "Cross-spectral coherence per field pair (— GT,  -- FFM)",
        fontsize=10,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------
def make_all_figures(
        out_dir: str | Path,
        xy: np.ndarray,
        basis: VizGraphBasis,
        fields_gt_repr: np.ndarray,
        fields_ffm_repr: np.ndarray,
        repo_out: Dict[str, object],
        names: Sequence[str],
        pair: Optional[Tuple[int, int]] = None,
        cfg: Optional[VizConfig] = None,
        prefix: str = "cross_spectral_coherence",
        ) -> Dict[str, Path]:
    cfg = cfg or VizConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ps.set_paper_style()

    C = fields_gt_repr.shape[1]
    pairs = field_pairs(C)

    coh_gt = repo_out["coh_target"]
    coh_ffm = repo_out["coh_pred"]
    Q_gt = repo_out["Q_target"]
    Q_ffm = repo_out["Q_pred"]

    gb_gt = band_pair_values_from_coherence(coh_gt, basis, pairs)
    gb_ffm = band_pair_values_from_coherence(coh_ffm, basis, pairs)

    if pair is None:
        pair_idx = int(np.argmax(gb_gt.mean(axis=0)))
        pair = pairs[pair_idx]
    else:
        pair = tuple(sorted(pair))
        pair_idx = pairs.index(pair)

    mat_gt = coherence_matrix(gb_gt, pairs, C)
    mat_ff = coherence_matrix(gb_ffm, pairs, C)
    mat_err = mat_ff - mat_gt

    loss_same = samefreq_loss_breakdown_from_coherence(
        coh_ffm,
        coh_gt,
        basis,
        pairs,
        cfg.n_freq_bins,
    )
    loss_xf = crossfreq_loss_breakdown_from_Q(Q_ffm, Q_gt, pairs)

    saved: Dict[str, Path] = {}

    fig = plt.figure(figsize=(13.0, 24.0))
    subs = fig.subfigures(5, 1, height_ratios=[1.05, 1.15, 0.85, 0.75, 1.15])

    draw_spatial_block(
        subs[0],
        xy,
        basis,
        fields_gt_repr,
        fields_ffm_repr,
        pair,
        names,
        gb_gt[:, pair_idx],
        gb_ffm[:, pair_idx],
        cfg,
        title=f"a   Scale-resolved coupling: {pair_label(names, *pair)}",
    )

    draw_matrix_block(
        subs[1],
        mat_gt,
        mat_err,
        names,
        title="b   Same-frequency cross-field coherence across all pairs, per scale",
    )

    draw_curve_block(
        subs[2],
        basis,
        coh_gt,
        coh_ffm,
        gb_gt,
        gb_ffm,
        pairs,
        pair_idx,
        pair,
        names,
        loss_same,
        cfg,
        title="c   Same-frequency coherence spectra and loss decomposition",
    )

    draw_crossfreq_block(
        subs[3],
        Q_gt,
        Q_ffm,
        pairs,
        pair,
        names,
        loss_xf,
        title=f"d   Cross-frequency band-energy coupling: {pair_label(names, *pair)}",
    )

    draw_crossfreq_spatial_block(
        subs[4],
        xy,
        basis,
        fields_gt_repr,
        fields_ffm_repr,
        Q_gt,
        Q_ffm,
        pair,
        names,
        cfg,
        title=f"e   Intuitive cross-frequency spatial coupling: {pair_label(names, *pair)}",
        top_k=4,
        mode="strongest_gt",
    )

    p = out_dir / f"{prefix}_combined.png"
    fig.savefig(p, dpi=cfg.dpi, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    saved["combined_png"] = p
    saved["combined_pdf"] = p.with_suffix(".pdf")

    fa = plt.figure(figsize=(13.0, 5.2))
    draw_spatial_block(
        fa.subfigures(1, 1),
        xy,
        basis,
        fields_gt_repr,
        fields_ffm_repr,
        pair,
        names,
        gb_gt[:, pair_idx],
        gb_ffm[:, pair_idx],
        cfg,
        title=f"Scale-resolved coupling: {pair_label(names, *pair)}",
    )
    pa = out_dir / f"{prefix}_panelA_spatial.png"
    fa.savefig(pa, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fa)
    saved["panelA"] = pa

    fb = plt.figure(figsize=(9.5, 6.2))
    draw_matrix_block(
        fb.subfigures(1, 1),
        mat_gt,
        mat_err,
        names,
        title="Same-frequency cross-field coherence across all pairs, per scale",
    )
    pb = out_dir / f"{prefix}_panelB_matrices.png"
    fb.savefig(pb, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fb)
    saved["panelB"] = pb

    fc = plt.figure(figsize=(11.5, 4.2))
    draw_curve_block(
        fc.subfigures(1, 1),
        basis,
        coh_gt,
        coh_ffm,
        gb_gt,
        gb_ffm,
        pairs,
        pair_idx,
        pair,
        names,
        loss_same,
        cfg,
        title="Same-frequency coherence spectra and loss decomposition",
    )
    pc = out_dir / f"{prefix}_panelC_curves.png"
    fc.savefig(pc, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fc)
    saved["panelC"] = pc

    fd = plt.figure(figsize=(12.5, 3.8))
    draw_crossfreq_block(
        fd.subfigures(1, 1),
        Q_gt,
        Q_ffm,
        pairs,
        pair,
        names,
        loss_xf,
        title=f"Cross-frequency band-energy coupling: {pair_label(names, *pair)}",
    )
    pd = out_dir / f"{prefix}_panelD_crossfreq.png"
    fd.savefig(pd, dpi=cfg.dpi, bbox_inches="tight")
    fd.savefig(pd.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fd)
    saved["panelD_crossfreq"] = pd
    saved["panelD_crossfreq_pdf"] = pd.with_suffix(".pdf")

    fe_spatial = plt.figure(figsize=(13.0, 5.8))
    draw_crossfreq_spatial_block(
        fe_spatial.subfigures(1, 1),
        xy,
        basis,
        fields_gt_repr,
        fields_ffm_repr,
        Q_gt,
        Q_ffm,
        pair,
        names,
        cfg,
        title=f"Intuitive cross-frequency spatial coupling: {pair_label(names, *pair)}",
        top_k=4,
        mode="strongest_gt",
    )
    pe_spatial = out_dir / f"{prefix}_panelE_crossfreq_spatial.png"
    fe_spatial.savefig(pe_spatial, dpi=cfg.dpi, bbox_inches="tight")
    fe_spatial.savefig(pe_spatial.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fe_spatial)
    saved["panelE_crossfreq_spatial"] = pe_spatial
    saved["panelE_crossfreq_spatial_pdf"] = pe_spatial.with_suffix(".pdf")

    fe = figure_all_pair_curves(basis, coh_gt, coh_ffm, pairs, names, cfg)
    pe = out_dir / f"{prefix}_allpairs_curves.png"
    fe.savefig(pe, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fe)
    saved["allpairs"] = pe

    return saved


# ---------------------------------------------------------------------------
# Extra spatial panels for additional field pairs
# ---------------------------------------------------------------------------
def make_pair_spatial_panels(
        out_dir: str | Path,
        xy: np.ndarray,
        basis: VizGraphBasis,
        fields_gt_repr: np.ndarray,
        fields_ffm_repr: np.ndarray,
        all_pairs: List[Tuple[int, int]],
        gb_gt: np.ndarray,
        gb_ffm: np.ndarray,
        names: Sequence[str],
        pairs: Sequence[Tuple[int, int]],
        cfg: Optional[VizConfig] = None,
        prefix: str = "cross_spectral_coherence",
        stacked: bool = True,
        ) -> Dict[str, Path]:
    cfg = cfg or VizConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ps.set_paper_style()
    pairs = [tuple(sorted(p)) for p in pairs]

    def _pidx(pair: Tuple[int, int]) -> int:
        try:
            return all_pairs.index(pair)
        except ValueError:
            raise ValueError(
                f"pair {pair} ({pair_label(names, *pair)}) not in all_pairs; "
                "field indices must be ascending and within range."
            )

    saved: Dict[str, Path] = {}

    for pair in pairs:
        idx = _pidx(pair)
        fig = plt.figure(figsize=(13.0, 5.2))
        draw_spatial_block(
            fig.subfigures(1, 1),
            xy,
            basis,
            fields_gt_repr,
            fields_ffm_repr,
            pair,
            names,
            gb_gt[:, idx],
            gb_ffm[:, idx],
            cfg,
            title=f"Scale-resolved coupling: {pair_label(names, *pair)}",
        )
        p = out_dir / f"{prefix}_pair_{names[pair[0]]}_{names[pair[1]]}_spatial.png"
        fig.savefig(p, dpi=cfg.dpi, bbox_inches="tight")
        plt.close(fig)
        saved[f"pair_{names[pair[0]]}_{names[pair[1]]}"] = p

    if stacked and len(pairs) > 1:
        fig = plt.figure(figsize=(13.0, 5.0 * len(pairs)))
        subs = fig.subfigures(len(pairs), 1)
        for sub, pair in zip(np.atleast_1d(subs), pairs):
            idx = _pidx(pair)
            draw_spatial_block(
                sub,
                xy,
                basis,
                fields_gt_repr,
                fields_ffm_repr,
                pair,
                names,
                gb_gt[:, idx],
                gb_ffm[:, idx],
                cfg,
                title=f"Scale-resolved coupling: {pair_label(names, *pair)}",
            )
        p = out_dir / f"{prefix}_multipair_spatial.png"
        fig.savefig(p, dpi=cfg.dpi, bbox_inches="tight")
        fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        saved["multipair_png"] = p
        saved["multipair_pdf"] = p.with_suffix(".pdf")

    return saved


# ===========================================================================
# CLI / evaluation (model checkpoint or offline npz)
# ===========================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Cross-spectral coherence figure for turbulent combustion."
    )

    # input selection
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Trained pointcloud-FFM run dir or .pt file (model mode).")
    p.add_argument("--data", type=str, default=None,
                   help="Override dataset .h5 path (else taken from run config).")
    p.add_argument("--split", type=str, default="val",
                   choices=["train", "val", "test"])
    p.add_argument("--npz-stack", type=str, nargs="*", default=None,
                   help="Offline mode: list of reconstruction .npz files.")

    # ensemble / conditioning
    p.add_argument("--ensemble-size", type=int, default=24,
                   help="Number of snapshots averaged for coherence estimation.")
    p.add_argument("--snapshot-indices", type=int, nargs="*", default=None,
                   help="Explicit split positions for the ensemble.")
    p.add_argument("--repr-index", type=int, default=None,
                   help="Split position used for the spatial maps.")
    p.add_argument("--cond-fields", type=int, nargs="*", default=None,
                   help="Conditioned field indices.")
    p.add_argument("--n-obs", type=int, default=256,
                   help="Sensors per conditioned field.")
    p.add_argument("--n-steps", type=int, default=50,
                   help="Generative ODE steps.")
    p.add_argument("--ode-solver", type=str, default=None)

    # graph-spectral basis
    p.add_argument("--n-modes", type=int, default=384,
                   help="Graph-Laplacian eigenmodes.")
    p.add_argument("--k-neighbors", type=int, default=12)
    p.add_argument("--sigma-scale", type=float, default=1.0)
    p.add_argument("--band-mode", type=str, default="thirds",
                   help="Kept for CLI compatibility; repo graph split currently uses thirds.")
    p.add_argument("--n-freq-bins", type=int, default=24,
                   help="Frequency bins for same-frequency coherence curves.")
    p.add_argument("--basis-cache", type=str, default=None,
                   help="Kept for CLI compatibility; not used by repo-backed basis adapter.")
    p.add_argument("--graph-basis", type=str, required=True,
                    help="Precomputed graph basis .pt from build_graph_basis.py.")
    

    # physical coherence
    p.add_argument("--eta-crossfreq", type=float, default=1.0,
                   help="Weight for cross-frequency coupling in L_phys_coh.")

    # figure
    p.add_argument("--pair", type=int, nargs=2, default=None,
                   help="Field indices for the primary spatial block.")
    p.add_argument("--extra-pairs", type=str, nargs="*", default=None,
                   help="Additional field pairs, e.g. --extra-pairs CO,T CH4,CO.")
    p.add_argument("--top-pairs", type=int, default=0,
                   help="Also render spatial panels for top-N most coherent GT pairs.")
    p.add_argument("--save-stack", type=str, default=None,
                   help="Directory to dump reconstructed ensemble as .npz files.")
    p.add_argument("--marker-size", type=float, default=2.2)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Pair parsing
# ---------------------------------------------------------------------------
def resolve_pair(token: str, names: Sequence[str]) -> Tuple[int, int]:
    parts = [t.strip() for t in token.replace(" ", "").split(",")]
    if len(parts) != 2:
        raise SystemExit(f"--extra-pairs expects 'a,b' tokens; got {token!r}.")

    def _idx(tok: str) -> int:
        if tok in names:
            return list(names).index(tok)
        try:
            i = int(tok)
        except ValueError:
            raise SystemExit(f"unknown field {tok!r}; choose from {list(names)}.")
        if not 0 <= i < len(names):
            raise SystemExit(f"field index {i} out of range [0, {len(names)}).")
        return i

    i, j = sorted((_idx(parts[0]), _idx(parts[1])))
    if i == j:
        raise SystemExit(f"--extra-pairs needs two distinct fields; got {token!r}.")
    return (i, j)


def save_stack_npz(
        stack_dir: Path,
        coords: np.ndarray,
        gt_stack: Sequence[np.ndarray],
        ffm_stack: Sequence[np.ndarray],
        names: Sequence[str],
        ) -> None:
    stack_dir.mkdir(parents=True, exist_ok=True)
    fn = np.asarray(list(names))
    for s, (g, f) in enumerate(zip(gt_stack, ffm_stack)):
        np.savez_compressed(
            stack_dir / f"snap_{s:03d}.npz",
            coords_xy=np.asarray(coords),
            truth_phys=np.asarray(g),
            recon_phys=np.asarray(f),
            field_names=fn,
        )
    print(
        f"[csc] saved reconstructed stack ({len(gt_stack)} snaps) -> "
        f"{stack_dir}  (re-render via --npz-stack '{stack_dir}/*.npz')"
    )


# ---------------------------------------------------------------------------
# Offline NPZ mode
# ---------------------------------------------------------------------------
def load_from_npz(paths: Sequence[str]) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], List[str]]:
    files = []
    for pat in paths:
        files.extend(
            sorted(Path(p) for p in glob.glob(pat))
            if any(c in pat for c in "*?[")
            else [Path(pat)]
        )

    if not files:
        raise FileNotFoundError(f"No .npz files matched: {paths}")

    coords = None
    names: List[str] = []
    gt, ffm = [], []

    for f in files:
        z = np.load(f, allow_pickle=True)
        c = np.asarray(z["coords_xy"], dtype=np.float64)
        if coords is None:
            coords = c
            names = [str(x) for x in z["field_names"]] if "field_names" in z else []
        gt.append(np.asarray(z["truth_phys"], dtype=np.float64))
        ffm.append(np.asarray(z["recon_phys"], dtype=np.float64))

    if not names:
        names = [f"f{c}" for c in range(gt[0].shape[1])]

    print(
        f"[csc] loaded {len(gt)} snapshots from npz; "
        f"N={coords.shape[0]}, fields={names}"
    )
    return coords, gt, ffm, names


# ---------------------------------------------------------------------------
# Model mode
# ---------------------------------------------------------------------------
def _reconcile_arch_with_checkpoint(cfg: dict, state_dict: dict) -> dict:
    cfg = dict(cfg)
    coord_dim = 3

    def _shape(name):
        for k in (name, f"model.{name}"):
            if k in state_dict:
                return tuple(state_dict[k].shape)
        return None

    fe = _shape("field_embed.weight")
    sp = _shape("sensor_in_proj.0.weight")

    if fe is not None:
        cfg["field_embed_dim"] = int(fe[1])

    if fe is not None and sp is not None:
        field_embed_dim = int(fe[1])
        sensor_coord_dim = int(sp[1]) - 1 - field_embed_dim

        if sensor_coord_dim == coord_dim:
            cfg["sensor_coord_encoding"] = "raw"
        elif sensor_coord_dim > coord_dim and sensor_coord_dim % (2 * coord_dim) == 0:
            cfg["sensor_coord_encoding"] = "fourier"
            cfg["use_fourier_pe"] = True
            cfg["pe_num_bands"] = sensor_coord_dim // (2 * coord_dim)

        print(
            f"[csc] checkpoint sensor_in_proj in_dim={sp[1]} -> "
            f"sensor_coord_encoding={cfg.get('sensor_coord_encoding')}, "
            f"use_fourier_pe={cfg.get('use_fourier_pe')}, "
            f"pe_num_bands={cfg.get('pe_num_bands')}"
        )

    return cfg


def load_model_and_dataset(args):
    import torch
    import evaluate_coherence as EC

    EC.set_seed(args.seed)

    checkpoint_path, run_dir = EC.choose_checkpoint(args.checkpoint)
    cfg = EC.load_run_config(run_dir)
    demo_dir = EC.infer_demo_dir(run_dir)

    device = torch.device(
        args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )

    baseline_model = str(cfg.get("baseline_model", "")).strip().lower() or None

    if baseline_model in {"latent_fm", "s3gm", "sit"}:
        raise NotImplementedError(
            f"This script currently targets pointcloud-FFM-style models; got "
            f"baseline_model={baseline_model!r}. Use evaluate_coherence.py for "
            "unified baselines, or extend load_model_and_dataset()."
        )

    data_path = args.data or cfg.get("data")
    if data_path is None:
        raise ValueError("Dataset path missing; pass --data.")

    data_path = EC.resolve_input_path(
        str(data_path),
        label="Dataset",
        extra_base_dirs=[run_dir, demo_dir],
    )

    stats_path = run_dir / "dataset_stats.pt"

    dataset = EC.TurbulentCombustionH5Dataset(
        h5_path=str(data_path),
        split=args.split,
        train_ratio=float(cfg.get("train_ratio", 0.9)),
        seed=int(cfg.get("seed", 42)),
        time_stride=int(cfg.get("time_stride", 1)),
        stats_path=str(stats_path) if stats_path.exists() else None,
    )

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except pickle.UnpicklingError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

    state_dict = (
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )

    if isinstance(state_dict, dict) and "_metadata" in state_dict:
        state_dict = {
            k: v for k, v in state_dict.items()
            if k != "_metadata"
        }

    cfg = _reconcile_arch_with_checkpoint(cfg, state_dict)

    model = EC.build_model(cfg, dataset).to(device)

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        print(f"[csc] strict load failed; retrying strict=False.\n  {str(exc)[:300]}")
        info = model.load_state_dict(state_dict, strict=False)

        n_miss = len(info.missing_keys)
        n_unexp = len(info.unexpected_keys)

        if n_miss or n_unexp:
            print(
                f"[csc] non-strict load: {n_miss} missing, {n_unexp} unexpected "
                "keys. Architecture may still differ from the checkpoint."
            )

    model.eval()

    if args.cond_fields is not None:
        cond_fields = list(args.cond_fields)
    else:
        cf = (
            cfg.get("vis_cond_fields")
            or cfg.get("cond_fields")
            or [cfg.get("cond_field", 2)]
        )
        cond_fields = EC.ensure_list(cf)

    # IMPORTANT:
    # This script builds a normal pointcloud model with EC.build_model(...).
    # Therefore it must use the pointcloud-FFM reconstruction path, not the
    # baseline reconstruction path. The baseline path expects model.adapter.
    recon_fn = EC.get_reconstruction_fn("pointcloud_ffm")

    print(f"[csc] built model type: {type(model).__name__}")
    print(f"[csc] using reconstruction function: {recon_fn.__name__}")
    print(f"[csc] conditioning fields: {cond_fields}")

    return model, dataset, device, cond_fields, recon_fn

def reconstruct_stack(
        model,
        dataset,
        device,
        recon_fn,
        positions,
        cond_fields,
        n_obs,
        n_steps,
        ode_solver,
        ):
    import torch

    sig = inspect.signature(recon_fn)
    gt, ffm = [], []

    for pos in positions:
        kwargs = dict(
            model=model,
            dataset=dataset,
            device=device,
            snapshot_index=int(pos),
            cond_fields=cond_fields,
            n_obs_list=[n_obs],
            n_steps=n_steps,
        )

        if "ode_solver" in sig.parameters:
            kwargs["ode_solver"] = ode_solver

        with torch.no_grad():
            out = recon_fn(**kwargs)

        t = out["truth"].detach().float().cpu().numpy()[0]
        r = out["recon"].detach().float().cpu().numpy()[0]

        gt.append(t)
        ffm.append(r)

        print(f"[csc] reconstructed split-position {pos} ({len(gt)}/{len(positions)})")

    coords = dataset.coords_raw.detach().cpu().numpy().astype(np.float64)
    return coords, gt, ffm, list(dataset.field_names)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def _as_float(x) -> float:
    return float(np.asarray(x).item())


def write_summary(
        path: Path,
        names,
        pairs,
        gb_gt,
        gb_ffm,
        loss_same,
        loss_xf,
        repo_out,
        selected_pair,
        basis,
        meta,
        ):
    band_names = [BAND_DISPLAY[k] for k in BAND_KEYS]
    plabels = [pair_label(names, i, j) for (i, j) in pairs]

    summary = {
        "selected_pair": pair_label(names, *selected_pair),
        "field_names": list(names),
        "n_modes": int(basis.n_modes),
        "band_edges_freq": [float(x) for x in basis.band_edges],
        "modes_per_band": {
            BAND_DISPLAY[k]: int(len(basis.bands[k]))
            for k in BAND_KEYS
        },
        "L_phys_coh": _as_float(repo_out["L_phys_coh"]),
        "L_same": _as_float(repo_out["L_same"]),
        "L_crossfreq": _as_float(repo_out["L_crossfreq"]),
        "samefreq_loss_total_plot": float(loss_same["total"]),
        "crossfreq_loss_total_plot": float(loss_xf["total"]),
        "per_pair": {},
        "meta": meta,
    }

    Q_gt = repo_out["Q_target"]
    Q_ffm = repo_out["Q_pred"]

    for p, lab in enumerate(plabels):
        i, j = pairs[p]

        summary["per_pair"][lab] = {
            "gamma2_gt": {
                band_names[b]: float(gb_gt[b, p])
                for b in range(3)
            },
            "gamma2_ffm": {
                band_names[b]: float(gb_ffm[b, p])
                for b in range(3)
            },
            "samefreq_loss": float(loss_same["per_pair"][p]),
            "crossfreq_loss": float(loss_xf["per_pair"][p]),
            "Q_gt": {
                band_names[a]: {
                    band_names[b]: float(Q_gt[a, b, i, j])
                    for b in range(3)
                }
                for a in range(3)
            },
            "Q_ffm": {
                band_names[a]: {
                    band_names[b]: float(Q_ffm[a, b, i, j])
                    for b in range(3)
                }
                for a in range(3)
            },
        }

    path.write_text(json.dumps(summary, indent=2))
    print(f"[csc] wrote summary -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.npz_stack:
        coords, gt_stack, ffm_stack, names = load_from_npz(args.npz_stack)
        out_root = Path(
            args.out_dir
            or (SRC_DIR.parent / "Save_PhyCoEval" / "cross_spectral_coherence" / ts)
        )
        meta = {"mode": "npz", "n_snapshots": len(gt_stack)}
    else:
        if not args.checkpoint:
            raise SystemExit("Provide --checkpoint (model mode) or --npz-stack.")

        model, dataset, device, cond_fields, recon_fn = load_model_and_dataset(args)

        if args.snapshot_indices:
            positions = list(args.snapshot_indices)
        else:
            m = min(args.ensemble_size, len(dataset))
            positions = sorted(
                set(np.linspace(0, len(dataset) - 1, m).astype(int).tolist())
            )

        coords, gt_stack, ffm_stack, names = reconstruct_stack(
            model,
            dataset,
            device,
            recon_fn,
            positions,
            cond_fields,
            args.n_obs,
            args.n_steps,
            args.ode_solver,
        )

        demo_dir = SRC_DIR.parent
        out_root = Path(
            args.out_dir
            or (demo_dir / "Save_PhyCoEval" / "cross_spectral_coherence" / ts)
        )

        meta = {
            "mode": "model",
            "checkpoint": str(args.checkpoint),
            "split": args.split,
            "cond_fields": list(cond_fields),
            "n_obs": args.n_obs,
            "n_steps": args.n_steps,
            "positions": [int(x) for x in positions],
        }

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.save_stack:
        save_stack_npz(Path(args.save_stack), coords, gt_stack, ffm_stack, names)

    repr_pos = 0 if args.repr_index is None else int(args.repr_index)
    repr_pos = min(repr_pos, len(gt_stack) - 1)
    fields_gt_repr = gt_stack[repr_pos]
    fields_ffm_repr = ffm_stack[repr_pos]

    # Graph-spectral basis from repo graph.py through helper adapter.
    basis = load_viz_graph_basis(
        args.graph_basis,
        current_coords=coords,
    )
    xy = coords[:, basis.keep_axes]

    # Same-frequency + cross-frequency diagnostics from repo cross_spectral.py.
    repo_out = compute_repo_outputs(
        basis,
        gt_stack,
        ffm_stack,
        eta_crossfreq=args.eta_crossfreq,
        device="cpu",
    )

    C = gt_stack[0].shape[1]
    pairs = field_pairs(C)

    coh_gt = repo_out["coh_target"]
    coh_ffm = repo_out["coh_pred"]
    Q_gt = repo_out["Q_target"]
    Q_ffm = repo_out["Q_pred"]

    gb_gt = band_pair_values_from_coherence(coh_gt, basis, pairs)
    gb_ffm = band_pair_values_from_coherence(coh_ffm, basis, pairs)

    loss_same = samefreq_loss_breakdown_from_coherence(
        coh_ffm,
        coh_gt,
        basis,
        pairs,
        args.n_freq_bins,
    )
    loss_xf = crossfreq_loss_breakdown_from_Q(Q_ffm, Q_gt, pairs)

    ranking = rank_pairs_by_band_coherence(gb_gt)
    print(
        "[csc] GT pair ranking (most coherent first): "
        + ", ".join(pair_label(names, *pairs[k]) for k in ranking[:5])
    )

    cfg = VizConfig(marker_size=args.marker_size, n_freq_bins=args.n_freq_bins)
    pair = tuple(sorted(args.pair)) if args.pair else None

    saved = make_all_figures(
        out_root,
        xy,
        basis,
        fields_gt_repr,
        fields_ffm_repr,
        repo_out,
        names,
        pair=pair,
        cfg=cfg,
    )

    saved.update(
        make_crossfreq_spatial_panels(
            out_root,
            xy,
            basis,
            fields_gt_repr,
            fields_ffm_repr,
            Q_gt,
            Q_ffm,
            names,
            pairs,   # this means ALL field pairs
            cfg=cfg,
            prefix="cross_spectral_coherence",
            stacked=True,
            top_k_bandpairs=4,
            mode="strongest_gt",
        )
    )

    selected = pair or pairs[int(np.argmax(gb_gt.mean(axis=0)))]

    extra: List[Tuple[int, int]] = []
    if args.extra_pairs:
        extra += [resolve_pair(tok, names) for tok in args.extra_pairs]

    if args.top_pairs > 0:
        extra += [tuple(sorted(pairs[int(k)])) for k in ranking[:args.top_pairs]]

    seen, extra_unique = set(), []
    for pr in extra:
        if pr not in seen:
            seen.add(pr)
            extra_unique.append(pr)

    if extra_unique:
        print(
            "[csc] extra spatial panels: "
            + ", ".join(pair_label(names, *pr) for pr in extra_unique)
        )
        saved.update(
            make_pair_spatial_panels(
                out_root,
                xy,
                basis,
                fields_gt_repr,
                fields_ffm_repr,
                pairs,
                gb_gt,
                gb_ffm,
                names,
                extra_unique,
                cfg=cfg,
            )
        )

    write_summary(
        out_root / "cross_spectral_coherence_summary.json",
        names,
        pairs,
        gb_gt,
        gb_ffm,
        loss_same,
        loss_xf,
        repo_out,
        selected,
        basis,
        meta,
    )

    print("\n[csc] done. Figures saved:")
    for k, v in saved.items():
        print(f"    {k:22s} {v}")


if __name__ == "__main__":
    main()