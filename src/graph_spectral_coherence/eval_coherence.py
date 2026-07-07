import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import numpy as np
import torch
import matplotlib.pyplot as plt
import json

from helpers import (
    TurbulentCombustionH5Dataset,
    reconstruct_snapshot,
)
from model_finetune import (
    load_pretrained_ffm,
    load_source_config,
)

from graph_spectral_coherence.graph_basis import make_graph_frequency_bands

from cross_spectral import (
    gft,
    compute_band_energy,
    compute_all_cross_band_covariances,
)

# -----------------------------------------------------------------------------
# Band Energy Diagnostics
# -----------------------------------------------------------------------------

def _safe_name(name):
    """Make a string safe for filenames."""
    return str(name).replace("/", "_").replace(" ", "_").replace("→", "to")

def _cross_spectral_coherence_band_metrics(
    fields_true,
    fields_pred,
    U,
    bands,
    field_pairs=None,
    eps: float = 1e-12,
):
    """
    Compute band-energy agreement diagnostics for both same-frequency and
    cross-frequency visualizations.

    SAME-FREQUENCY BAND ENERGY:
        For each graph-frequency band m and field c:

            E_same[m, c] = mean_b sum_{k in band m} |x_hat[b, k, c]|^2

        Ratio:

            E_same_pred[m, c] / E_same_true[m, c]

    CROSS-FREQUENCY BAND ENERGY:
        Uses the raw cross-band covariance object S from the cross-frequency
        coherence code:

            S[m, n, c1, c2]

        where S measures batch co-fluctuation between centered band energy of
        field c1 in band m and centered band energy of field c2 in band n.

        Ratio:

            |S_pred[m, n, c1, c2]| / |S_true[m, n, c1, c2]|

    Args:
        fields_true: [B, N, C]
        fields_pred: [B, N, C]
        U: [N, K] graph Fourier basis
        bands: dict mapping band name -> graph-frequency indices
        field_pairs: optional list of (c1, c2). If None, uses all c1 < c2.
        eps: numerical stability

    Returns:
        metrics: flat scalar dict
        payload: arrays for plotting
    """

    # ---------------------------------------------------
    # 1. Graph Fourier transform
    # ---------------------------------------------------
    gft_true = gft(fields_true, U)  # [B, K, C]
    gft_pred = gft(fields_pred, U)  # [B, K, C]

    device = gft_true.device
    C = gft_true.shape[-1]

    if field_pairs is None:
        field_pairs = [(i, j) for i in range(C) for j in range(i + 1, C)]

    band_names = list(bands.keys())
    num_bands = len(band_names)
    num_pairs = len(field_pairs)

    # ---------------------------------------------------
    # 2. Same-frequency band energy
    # ---------------------------------------------------
    samefreq_energy_true = []
    samefreq_energy_pred = []

    for band_name in band_names:
        band_idx = bands[band_name]

        # [B, C]
        E_true_bc = compute_band_energy(gft_true, band_idx)
        E_pred_bc = compute_band_energy(gft_pred, band_idx)

        # Average over batch -> [C]
        E_true_c = E_true_bc.mean(dim=0)
        E_pred_c = E_pred_bc.mean(dim=0)

        samefreq_energy_true.append(E_true_c)
        samefreq_energy_pred.append(E_pred_c)

    # [M, C]
    samefreq_energy_true = torch.stack(samefreq_energy_true, dim=0)
    samefreq_energy_pred = torch.stack(samefreq_energy_pred, dim=0)

    # [M, C]
    samefreq_energy_ratio = samefreq_energy_pred / (samefreq_energy_true + eps)

    samefreq_energy_relerr = torch.abs(samefreq_energy_pred - samefreq_energy_true) / (samefreq_energy_true + eps)

    # ---------------------------------------------------
    # 3. Cross-frequency band energy
    # ---------------------------------------------------
    # S has shape [M, M, C, C].
    # S[m, n, c1, c2] measures cross-band energy co-fluctuation.
    S_true, band_names_from_S = compute_all_cross_band_covariances(
        gft_true,
        bands,
    )
    S_pred, _ = compute_all_cross_band_covariances(
        gft_pred,
        bands,
    )

    # Use the band ordering returned by the cross-band covariance function.
    band_names = list(band_names_from_S)
    num_bands = len(band_names)

    crossfreq_energy_true = torch.empty(
        (num_bands, num_bands, num_pairs),
        device=device,
        dtype=S_true.real.dtype,
    )

    crossfreq_energy_pred = torch.empty_like(crossfreq_energy_true)

    for p, (c1, c2) in enumerate(field_pairs):
        crossfreq_energy_true[:, :, p] = torch.abs(S_true[:, :, c1, c2])
        crossfreq_energy_pred[:, :, p] = torch.abs(S_pred[:, :, c1, c2])

    # [M, M, P]
    crossfreq_energy_ratio = crossfreq_energy_pred / (crossfreq_energy_true + eps)

    crossfreq_energy_relerr = torch.abs(crossfreq_energy_pred - crossfreq_energy_true) / (crossfreq_energy_true + eps)

    # Cross-frequency means off-diagonal band pairs only.
    off_diag_mask = ~torch.eye(num_bands, dtype=torch.bool, device=device)

    crossfreq_ratio_offdiag = crossfreq_energy_ratio[off_diag_mask, :]
    crossfreq_relerr_offdiag = crossfreq_energy_relerr[off_diag_mask, :]

    # ---------------------------------------------------
    # 4. Scalar metrics
    # ---------------------------------------------------
    metrics = {}

    # Same-frequency summaries.
    samefreq_energy_ratio_mean_by_band = samefreq_energy_ratio.mean(dim=1)
    samefreq_energy_relerr_mean_by_band = samefreq_energy_relerr.mean(dim=1)

    for m, band_name in enumerate(band_names):
        clean = str(band_name).lower()

        metrics[f"samefreq_energy_ratio_{clean}"] = float(
            samefreq_energy_ratio_mean_by_band[m].detach().cpu()
        )

        metrics[f"samefreq_energy_relerr_{clean}"] = float(
            samefreq_energy_relerr_mean_by_band[m].detach().cpu()
        )

    metrics["samefreq_energy_ratio_mean"] = float(
        samefreq_energy_ratio.mean().detach().cpu()
    )

    metrics["samefreq_energy_relerr_mean"] = float(
        samefreq_energy_relerr.mean().detach().cpu()
    )

    # Cross-frequency summaries.
    metrics["crossfreq_energy_ratio_mean"] = float(
        crossfreq_ratio_offdiag.mean().detach().cpu()
    )

    metrics["crossfreq_energy_relerr_mean"] = float(
        crossfreq_relerr_offdiag.mean().detach().cpu()
    )

    for i, band_i in enumerate(band_names):
        for j, band_j in enumerate(band_names):
            if i == j:
                continue

            key = f"{str(band_i).lower()}_to_{str(band_j).lower()}"

            metrics[f"crossfreq_energy_ratio_{key}"] = float(
                crossfreq_energy_ratio[i, j, :].mean().detach().cpu()
            )

            metrics[f"crossfreq_energy_relerr_{key}"] = float(
                crossfreq_energy_relerr[i, j, :].mean().detach().cpu()
            )

    # ---------------------------------------------------
    # 5. Payload for plotting / saving
    # ---------------------------------------------------
    payload = {
        "band_names": np.asarray(band_names),
        "field_pairs": np.asarray(field_pairs),

        # Same-frequency band energy, shape [M, C].
        "samefreq_energy_true": samefreq_energy_true.detach().cpu().numpy(),
        "samefreq_energy_pred": samefreq_energy_pred.detach().cpu().numpy(),
        "samefreq_energy_ratio": samefreq_energy_ratio.detach().cpu().numpy(),
        "samefreq_energy_relerr": samefreq_energy_relerr.detach().cpu().numpy(),

        # Cross-frequency band energy, shape [M, M, P].
        "crossfreq_energy_true": crossfreq_energy_true.detach().cpu().numpy(),
        "crossfreq_energy_pred": crossfreq_energy_pred.detach().cpu().numpy(),
        "crossfreq_energy_ratio": crossfreq_energy_ratio.detach().cpu().numpy(),
        "crossfreq_energy_relerr": crossfreq_energy_relerr.detach().cpu().numpy(),

        # Raw cross-band covariance tensors, shape [M, M, C, C].
        "S_true": S_true.detach().cpu().numpy(),
        "S_pred": S_pred.detach().cpu().numpy(),
    }

    return metrics, payload

def _cross_spectral_coherence_band_metrics_snapshot(
    fields_true,
    fields_pred,
    U,
    bands,
    field_pairs=None,
    eps: float = 1e-12,
):
    """
    Per-snapshot band-energy diagnostics only.

    Computes same-frequency and cross-frequency band-energy relative erros per snapshot.

    Args:
        fields_true: [B, N, C]
        fields_pred: [B, N, C]
        U: [N, K] graph fourier basis
        bands: dict mapping band name -> graph frequency indices
        field_pairs: optional list of (c1, c2). If None, uses all c1 < c2.
        eps: numerical stability
    
    Returns:
        metrics:
    """

    # ---------------------------------------------------
    # 1. Graph Fourier Transform
    # ---------------------------------------------------
    gft_true = gft(fields_true, U)
    gft_pred = gft(fields_pred, U)

    device = gft_true.device
    B, _, C = gft_true.shape

    if field_pairs is None:
        field_pairs = [(i, j) for i in range(C) for j in range(i + 1, C)]
    
    band_names = list(bands.keys())
    M = len(band_names)
    P = len(field_pairs)

    # ---------------------------------------------------
    # 2. Same-frequency band energy per snapshot
    # ---------------------------------------------------
    samefreq_energy_true = []
    samefreq_energy_pred = []

    for bandName in band_names:
        band_idx = bands[bandName]

        E_true_bc = compute_band_energy(gft_true, band_idx)
        E_pred_bc = compute_band_energy(gft_pred, band_idx)

        # [B, C]
        samefreq_energy_true.append(E_true_bc)
        samefreq_energy_pred.append(E_pred_bc)
    
    # [B, M, C]
    samefreq_energy_true = torch.stack(samefreq_energy_true, dim=1)
    samefreq_energy_pred = torch.stack(samefreq_energy_pred, dim=1)

    samefreq_energy_ratio = samefreq_energy_pred / (samefreq_energy_true + eps)

    samefreq_energy_relerr = torch.abs(samefreq_energy_pred - samefreq_energy_true) / (samefreq_energy_true + eps)

    # [B]
    samefreq_relerr_per_snapshot = samefreq_energy_relerr.mean(dim=(1,2))
    
    # ---------------------------------------------------
    # 3. Cross-frequency band energy per snapshot
    # ---------------------------------------------------
    # Center band energies across batch, matching your cross-frequency
    # covariance construction but keeping per-snapshot products instead
    # of immediately averaging over B.
    z_true = samefreq_energy_true - samefreq_energy_true.mean(dim=0, keepdim=True)
    z_pred = samefreq_energy_pred - samefreq_energy_pred.mean(dim=0, keepdim=True)

    crossfreq_energy_true = torch.empty(
        (B, M, M, P),
        device=device,
        dtype=z_true.dtype,
    )

    crossfreq_energy_pred = torch.empty_like(crossfreq_energy_true)

    for p, (c1, c2) in enumerate(field_pairs):
        # [B, M, M]
        true_pair = torch.einsum(
            "bm,bn->bmn",
            z_true[:, :, c1],
            z_true[:, :, c2],
        )

        pred_pair = torch.einsum(
            "bm,bn->bmn",
            z_pred[:, :, c1],
            z_pred[:, :, c2],
        )

        crossfreq_energy_true[:, :, :, p] = torch.abs(true_pair)
        crossfreq_energy_pred[:, :, :, p] = torch.abs(pred_pair)

    crossfreq_energy_ratio = crossfreq_energy_pred / (crossfreq_energy_true + eps)

    crossfreq_energy_relerr = torch.abs(crossfreq_energy_pred - crossfreq_energy_true) / (crossfreq_energy_true + eps)

    # Cross-frequency = off-diagonal band pairs only.
    off_diag_mask = ~torch.eye(M, dtype=torch.bool, device=device)

    # [B, M*(M-1), P]
    crossfreq_relerr_offdiag = crossfreq_energy_relerr[:, off_diag_mask, :]

    # [B]
    crossfreq_relerr_per_snapshot = crossfreq_relerr_offdiag.mean(dim=(1, 2))

    # ---------------------------------------------------
    # 4. Total error per snapshot, still not a reward
    # ---------------------------------------------------
    total_relerr_per_snapshot = (samefreq_relerr_per_snapshot + crossfreq_relerr_per_snapshot)

    # ---------------------------------------------------
    # 5. Scalar logging metrics
    # ---------------------------------------------------
    metrics = {
        "samefreq_relerr_snapshot_mean": float(
            samefreq_relerr_per_snapshot.mean().detach().cpu()
        ),
        "samefreq_relerr_snapshot_std": float(
            samefreq_relerr_per_snapshot.std(unbiased=False).detach().cpu()
        ),
        "crossfreq_relerr_snapshot_mean": float(
            crossfreq_relerr_per_snapshot.mean().detach().cpu()
        ),
        "crossfreq_relerr_snapshot_std": float(
            crossfreq_relerr_per_snapshot.std(unbiased=False).detach().cpu()
        ),
        "total_relerr_snapshot_mean": float(
            total_relerr_per_snapshot.mean().detach().cpu()
        ),
        "total_relerr_snapshot_std": float(
            total_relerr_per_snapshot.std(unbiased=False).detach().cpu()
        ),
    }

    # Same-frequency per-band means.
    samefreq_relerr_band_mean = samefreq_energy_relerr.mean(dim=(0, 2))  # [M]

    for m, band_name in enumerate(band_names):
        clean = str(band_name).lower()
        metrics[f"samefreq_relerr_{clean}_snapshot_mean"] = float(
            samefreq_relerr_band_mean[m].detach().cpu()
        )

    # Cross-frequency per-band-pair means.
    for i, band_i in enumerate(band_names):
        for j, band_j in enumerate(band_names):
            if i == j:
                continue

            key = f"{str(band_i).lower()}_to_{str(band_j).lower()}"

            metrics[f"crossfreq_relerr_{key}_snapshot_mean"] = float(
                crossfreq_energy_relerr[:, i, j, :].mean().detach().cpu()
            )

    # ---------------------------------------------------
    # 6. Payload
    # ---------------------------------------------------
    payload = {
        "band_names": np.asarray(band_names),
        "field_pairs": np.asarray(field_pairs),

        # Same-frequency per-snapshot energy, [B, M, C].
        "samefreq_energy_true": samefreq_energy_true.detach().cpu().numpy(),
        "samefreq_energy_pred": samefreq_energy_pred.detach().cpu().numpy(),
        "samefreq_energy_ratio": samefreq_energy_ratio.detach().cpu().numpy(),
        "samefreq_energy_relerr": samefreq_energy_relerr.detach().cpu().numpy(),

        # Cross-frequency per-snapshot energy contribution, [B, M, M, P].
        "crossfreq_energy_true": crossfreq_energy_true.detach().cpu().numpy(),
        "crossfreq_energy_pred": crossfreq_energy_pred.detach().cpu().numpy(),
        "crossfreq_energy_ratio": crossfreq_energy_ratio.detach().cpu().numpy(),
        "crossfreq_energy_relerr": crossfreq_energy_relerr.detach().cpu().numpy(),

        # Per-snapshot error pieces, [B].
        "samefreq_relerr_per_snapshot": (
            samefreq_relerr_per_snapshot.detach().cpu().numpy()
        ),
        "crossfreq_relerr_per_snapshot": (
            crossfreq_relerr_per_snapshot.detach().cpu().numpy()
        ),
        "total_relerr_per_snapshot": (
            total_relerr_per_snapshot.detach().cpu().numpy()
        ),
    }

    return metrics, payload

# -----------------------------------------------------------------------------
# Band-Energy Ratio Plots
# -----------------------------------------------------------------------------
EPS = 1e-12

BAND_KEYS = ("low", "mid", "high")

# Low graph frequency corresponds to large spatial scales.
BAND_DISPLAY = {
    "low": "large",
    "mid": "medium",
    "high": "small",
}

BAND_TINT = (
    "#e8e8f7",  # large scale
    "#e7f2e7",  # medium scale
    "#f8e7e7",  # small scale
)

BAND_EDGE_C = (
    "#5b5bd6",
    "#2e8b57",
    "#c0392b",
)

_OFFDIAG_LABELS = [
    ("low", "mid", "Low→Mid"),
    ("low", "high", "Low→High"),
    ("mid", "low", "Mid→Low"),
    ("mid", "high", "Mid→High"),
    ("high", "low", "High→Low"),
    ("high", "mid", "High→Mid"),
]


def _ordered_band_indices(band_names):
    """
    Map low/mid/high to their positions in the supplied band ordering.
    """
    names = [
        str(name).lower()
        for name in np.asarray(band_names)
    ]

    missing = [
        band_name
        for band_name in BAND_KEYS
        if band_name not in names
    ]

    if missing:
        raise ValueError(
            "Expected low, mid, and high frequency bands. "
            f"Missing={missing}; received={names}."
        )

    return {
        band_name: names.index(band_name)
        for band_name in BAND_KEYS
    }


def _save_samefreq_band_energy_ratio_plot(
    band_names,
    samefreq_energy_true,
    samefreq_energy_pred,
    save_path,
    title="Per-Field Band Energy Ratio (Prediction / Ground Truth)",
    field_names=None,
):
    """
    Match visualization.py Panel F.

    Makes one subplot for each spatial scale:
        large  = low graph frequency
        medium = mid graph frequency
        small  = high graph frequency

    Each bar is:

        E_pred[b, c] / E_GT[b, c]
    """
    be_gt = np.asarray(
        samefreq_energy_true,
        dtype=np.float64,
    )

    be_pred = np.asarray(
        samefreq_energy_pred,
        dtype=np.float64,
    )

    if be_gt.shape != be_pred.shape:
        raise ValueError(
            "GT and prediction energy arrays must have the same shape. "
            f"Got GT={be_gt.shape}, pred={be_pred.shape}."
        )

    if be_gt.ndim != 2:
        raise ValueError(
            "Expected same-frequency energy arrays with shape [M,C]. "
            f"Got {be_gt.shape}."
        )

    _, num_fields = be_gt.shape

    if field_names is None:
        field_names = [
            f"field_{field_index}"
            for field_index in range(num_fields)
        ]

    band_idx = _ordered_band_indices(band_names)

    x = np.arange(num_fields)
    width = 0.65

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.0, 4.2),
    )

    for position, band_name in enumerate(BAND_KEYS):
        ax = axes[position]

        row = band_idx[band_name]

        ratio = (
            be_pred[row]
            / (be_gt[row] + EPS)
        )

        bars = ax.bar(
            x,
            ratio,
            width=width,
            color=BAND_TINT[position],
            edgecolor=BAND_EDGE_C[position],
            linewidth=0.7,
        )

        ax.axhline(
            1.0,
            color="black",
            linewidth=0.9,
            linestyle="--",
            zorder=5,
        )

        ax.set_xticks(x)

        ax.set_xticklabels(
            list(field_names),
            rotation=40,
            ha="right",
            fontsize=7.5,
        )

        ax.set_ylabel(
            r"$E^{\mathrm{pred}}_b / E^{\mathrm{GT}}_b$",
            fontsize=8.5,
        )

        ax.set_title(
            f"{BAND_DISPLAY[band_name].capitalize()} scale",
            fontsize=9,
            color=BAND_EDGE_C[position],
            fontweight="bold",
        )

        ax.set_ylim(
            0,
            max(
                1.6,
                float(np.max(ratio)) * 1.15,
            ),
        )

        ax.yaxis.grid(
            True,
            linewidth=0.4,
            alpha=0.6,
        )

        ax.set_axisbelow(True)

        for bar, value in zip(bars, ratio):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                float(value) + 0.03,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=6.2,
            )

    fig.suptitle(
        title,
        fontsize=10.5,
        fontweight="bold",
        y=0.98,
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.94)
    )

    save_path = Path(save_path)

    fig.savefig(
        save_path,
        dpi=400,
        bbox_inches="tight",
    )

    fig.savefig(
        save_path.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    plt.close(fig)


def _crossband_ratio_values(
    band_names,
    S_true,
    S_pred,
    pair,
):
    """
    Return the six directed cross-band ratios used by visualization.py.
    """
    S_gt = np.asarray(S_true)
    S_model = np.asarray(S_pred)

    if S_gt.shape != S_model.shape:
        raise ValueError(
            "GT and prediction S arrays must have the same shape. "
            f"Got GT={S_gt.shape}, pred={S_model.shape}."
        )

    if S_gt.ndim != 4:
        raise ValueError(
            "Expected S arrays with shape [M,M,C,C]. "
            f"Got {S_gt.shape}."
        )

    c1, c2 = map(int, pair)

    band_idx = _ordered_band_indices(
        band_names
    )

    ratios = []
    labels = []
    colors = []
    edge_colors = []

    for (
        source_band,
        target_band,
        display_label,
    ) in _OFFDIAG_LABELS:

        source_index = band_idx[source_band]
        target_index = band_idx[target_band]

        gt_value = float(
            np.abs(
                S_gt[
                    source_index,
                    target_index,
                    c1,
                    c2,
                ]
            )
        )

        pred_value = float(
            np.abs(
                S_model[
                    source_index,
                    target_index,
                    c1,
                    c2,
                ]
            )
        )

        ratios.append(
            pred_value / (gt_value + EPS)
        )

        labels.append(display_label)

        source_position = BAND_KEYS.index(
            source_band
        )

        colors.append(
            BAND_TINT[source_position]
        )

        edge_colors.append(
            BAND_EDGE_C[source_position]
        )

    return (
        np.asarray(ratios, dtype=np.float64),
        labels,
        colors,
        edge_colors,
    )


def _draw_crossfreq_ratio_axis(
    ax,
    band_names,
    S_true,
    S_pred,
    pair,
    field_names=None,
):
    """
    Draw one visualization.py-style Panel H on an existing axis.
    """
    c1, c2 = map(int, pair)

    (
        ratios,
        labels,
        colors,
        edge_colors,
    ) = _crossband_ratio_values(
        band_names=band_names,
        S_true=S_true,
        S_pred=S_pred,
        pair=(c1, c2),
    )

    x = np.arange(len(ratios))

    bars = ax.bar(
        x,
        ratios,
        color=colors,
        edgecolor=edge_colors,
        linewidth=0.8,
        width=0.62,
    )

    ax.axhline(
        1.0,
        color="black",
        linewidth=1.0,
        linestyle="--",
        zorder=5,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        labels,
        rotation=35,
        ha="right",
        fontsize=8.5,
    )

    ax.set_ylabel(
        r"$|S^{\mathrm{pred}}| / |S^{\mathrm{GT}}|$",
        fontsize=9,
    )

    ax.set_xlim(
        -0.5,
        len(ratios) - 0.5,
    )

    ax.set_ylim(
        0,
        max(
            1.6,
            float(np.max(ratios)) * 1.18,
        ),
    )

    ax.yaxis.grid(
        True,
        linewidth=0.4,
        alpha=0.6,
    )

    ax.set_axisbelow(True)

    for bar, value in zip(bars, ratios):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            float(value) + 0.04,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=BAND_TINT[position],
            ec=BAND_EDGE_C[position],
            lw=0.7,
        )
        for position in range(3)
    ]

    ax.legend(
        handles,
        [
            f"Source: {BAND_DISPLAY[band].capitalize()}"
            for band in BAND_KEYS
        ],
        fontsize=7.0,
        loc="upper right",
        framealpha=0.85,
    )

    if field_names is None:
        pair_name = f"field{c1}–field{c2}"
    else:
        pair_name = (
            f"{field_names[c1]}–"
            f"{field_names[c2]}"
        )

    ax.set_title(
        pair_name,
        fontsize=9,
        fontweight="bold",
    )


def _save_crossfreq_band_energy_ratio_plot(
    band_names,
    S_true,
    S_pred,
    pair,
    save_path,
    field_names=None,
    title="Cross-Frequency Band Energy Ratio",
):
    """
    Save one cross-frequency ratio figure for one physical-field pair.
    """
    fig, ax = plt.subplots(
        figsize=(9.5, 3.8)
    )

    _draw_crossfreq_ratio_axis(
        ax=ax,
        band_names=band_names,
        S_true=S_true,
        S_pred=S_pred,
        pair=pair,
        field_names=field_names,
    )

    fig.suptitle(
        title,
        fontsize=10.5,
        fontweight="bold",
        y=0.97,
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.92)
    )

    save_path = Path(save_path)

    fig.savefig(
        save_path,
        dpi=400,
        bbox_inches="tight",
    )

    fig.savefig(
        save_path.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    plt.close(fig)


def _save_stacked_crossfreq_band_energy_ratio_plot(
    band_names,
    S_true,
    S_pred,
    field_pairs,
    save_path,
    field_names=None,
):
    """
    Save every physical-field pair in one vertically stacked figure.

    This does not average ratios across field pairs.
    Each field pair keeps its own six directed bars.
    """
    pairs = [
        tuple(map(int, pair))
        for pair in np.asarray(field_pairs)
    ]

    num_pairs = len(pairs)

    fig, axes = plt.subplots(
        num_pairs,
        1,
        figsize=(9.5, 3.6 * num_pairs),
        squeeze=False,
    )

    for row, pair in enumerate(pairs):
        _draw_crossfreq_ratio_axis(
            ax=axes[row, 0],
            band_names=band_names,
            S_true=S_true,
            S_pred=S_pred,
            pair=pair,
            field_names=field_names,
        )

    fig.suptitle(
        "Cross-Frequency Band Energy Ratios for All Field Pairs",
        fontsize=11,
        fontweight="bold",
        y=0.995,
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.985)
    )

    save_path = Path(save_path)

    fig.savefig(
        save_path,
        dpi=400,
        bbox_inches="tight",
    )

    fig.savefig(
        save_path.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    plt.close(fig)


def save_band_energy_diagnostic_plots(
    payload,
    save_dir,
    field_names=None,
    save_per_pair=True,
):
    """
    Save visualization.py-style same-frequency and cross-frequency
    band-energy ratio plots.
    """
    save_dir = Path(save_dir)

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    band_names = payload["band_names"]

    field_pairs = [
        tuple(map(int, pair))
        for pair in np.asarray(
            payload["field_pairs"]
        )
    ]

    # ---------------------------------------------------------
    # Same-frequency ratio plot
    # ---------------------------------------------------------
    _save_samefreq_band_energy_ratio_plot(
        band_names=band_names,
        samefreq_energy_true=payload[
            "samefreq_energy_true"
        ],
        samefreq_energy_pred=payload[
            "samefreq_energy_pred"
        ],
        save_path=(
            save_dir
            / "samefreq_band_energy_ratio.png"
        ),
        field_names=field_names,
    )

    # ---------------------------------------------------------
    # One cross-frequency ratio plot per physical-field pair
    # ---------------------------------------------------------
    if save_per_pair:
        for c1, c2 in field_pairs:

            if field_names is None:
                pair_name = (
                    f"field{c1}_field{c2}"
                )
            else:
                pair_name = (
                    f"{field_names[c1]}_"
                    f"{field_names[c2]}"
                )

            _save_crossfreq_band_energy_ratio_plot(
                band_names=band_names,
                S_true=payload["S_true"],
                S_pred=payload["S_pred"],
                pair=(c1, c2),
                save_path=(
                    save_dir
                    / (
                        "crossfreq_band_energy_ratio_"
                        f"{_safe_name(pair_name)}.png"
                    )
                ),
                field_names=field_names,
            )

    # ---------------------------------------------------------
    # Stacked figure containing all field pairs
    # ---------------------------------------------------------
    if len(field_pairs) > 1:
        _save_stacked_crossfreq_band_energy_ratio_plot(
            band_names=band_names,
            S_true=payload["S_true"],
            S_pred=payload["S_pred"],
            field_pairs=field_pairs,
            save_path=(
                save_dir
                / "crossfreq_band_energy_ratio_all_pairs.png"
            ),
            field_names=field_names,
        )

def main():
    # eval_coherence.py is assumed to be inside:
    # 0_demo_TurbulentCombustion/src/cross_spectral/
    project_root = Path(__file__).resolve().parents[2]

    # -------------------------------------------------------------------------
    # 1. Paths and evaluation settings
    # -------------------------------------------------------------------------
    run_dir = (
        project_root
        / "Save_TrainedModel"
        / "ffm_tc_pointcloud_DemoN30_20260610_173403"
    )

    graph_basis_path = (
        project_root
        / "Save_Graph"
        / "graph_basis_k16_modes384.pt"
    )

    checkpoint_name = "best"
    checkpoint_path = run_dir / "best.pt"

    if not run_dir.exists():
        raise FileNotFoundError(
            f"Training run directory not found: {run_dir}"
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Best trained checkpoint not found: {checkpoint_path}"
        )

    if not graph_basis_path.exists():
        raise FileNotFoundError(
            f"Graph basis not found: {graph_basis_path}\n"
            "The graph basis must be the saved .pt file."
        )

    # Save only band-energy outputs in this dedicated folder.
    output_dir = (
        run_dir
        / "Evaluation"
        / "BandEnergyPlots"
        / checkpoint_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    split = "test"

    # Match visualization.py's default ensemble size.
    num_snapshots = 24

    # False means compute energy in the normalized model space.
    use_denorm = False

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    torch.manual_seed(42)
    np.random.seed(42)

    print(f"Run directory: {run_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Graph basis: {graph_basis_path}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")

    # -------------------------------------------------------------------------
    # 2. Load the saved training configuration
    # -------------------------------------------------------------------------
    source_cfg = load_source_config(run_dir)

    data_path = Path(
        source_cfg.get(
            "data",
            "Dataset/Merged_CH4COTU1P.h5",
        )
    )

    if not data_path.is_absolute():
        data_path = project_root / data_path

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    # -------------------------------------------------------------------------
    # 3. Load the dataset with this model's saved normalization statistics
    # -------------------------------------------------------------------------
    stats_path = run_dir / "dataset_stats.pt"

    if not stats_path.exists():
        raise FileNotFoundError(
            f"Dataset statistics not found: {stats_path}"
        )

    dataset = TurbulentCombustionH5Dataset(
        h5_path=str(data_path),
        split=split,
        train_ratio=float(
            source_cfg.get("train_ratio", 0.9)
        ),
        seed=int(
            source_cfg.get("seed", 42)
        ),
        time_stride=int(
            source_cfg.get("time_stride", 1)
        ),
        stats_path=str(stats_path),
    )

    field_names = list(dataset.field_names)

    print(f"Evaluation split: {split}")
    print(f"Dataset snapshots: {len(dataset)}")
    print(f"Number of points: {dataset.num_points}")
    print(f"Physical fields: {field_names}")

    # -------------------------------------------------------------------------
    # 4. Read conditioning settings from the saved training configuration
    # -------------------------------------------------------------------------
    cond_fields = (
        source_cfg.get("vis_cond_fields")
        or source_cfg.get("cond_fields")
    )

    if cond_fields is None:
        cond_fields = [
            int(
                source_cfg.get("cond_field")
                or 2
            )
        ]
    elif isinstance(
        cond_fields,
        (int, np.integer),
    ):
        cond_fields = [int(cond_fields)]
    else:
        cond_fields = [
            int(value)
            for value in cond_fields
        ]

    n_obs_list = (
        source_cfg.get("vis_n_obs_list")
        or source_cfg.get("n_obs_max_list")
    )

    if n_obs_list is None:
        n_obs_list = [
            int(
                source_cfg.get("n_obs_max")
                or 256
            )
        ]
    elif isinstance(
        n_obs_list,
        (int, np.integer),
    ):
        n_obs_list = [int(n_obs_list)]
    else:
        n_obs_list = [
            int(value)
            for value in n_obs_list
        ]

    # One observation count can be shared across all conditioned fields.
    if (
        len(n_obs_list) == 1
        and len(cond_fields) > 1
    ):
        n_obs_list = (
            n_obs_list
            * len(cond_fields)
        )

    if len(n_obs_list) != len(cond_fields):
        raise ValueError(
            "n_obs_list must have length 1 or match cond_fields. "
            f"Got cond_fields={cond_fields}, "
            f"n_obs_list={n_obs_list}."
        )

    n_steps_generation = int(
        source_cfg.get("n_steps_generation")
        or 100
    )

    ode_solver = str(
        source_cfg.get("ode_solver")
        or "euler"
    ).lower()

    if ode_solver not in {"euler", "heun"}:
        raise ValueError(
            f"Unsupported ode_solver={ode_solver!r}; "
            "expected 'euler' or 'heun'."
        )

    print(f"Conditioned fields: {cond_fields}")
    print(f"Observation counts: {n_obs_list}")
    print(f"Generation steps: {n_steps_generation}")
    print(f"ODE solver: {ode_solver}")

    # -------------------------------------------------------------------------
    # 5. Load the best trained checkpoint
    # -------------------------------------------------------------------------
    model, source_cfg_loaded, checkpoint = load_pretrained_ffm(
        source_run_dir=run_dir,
        checkpoint=checkpoint_name,
        dataset=dataset,
        device=device,
    )

    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")

    # -------------------------------------------------------------------------
    # 6. Force the PyTorch neighbor backend instead of KeOps
    # -------------------------------------------------------------------------
    outer_model = (
        model.module
        if hasattr(model, "module")
        else model
    )

    if not hasattr(outer_model, "model"):
        raise AttributeError(
            f"Loaded model {type(outer_model).__name__} "
            "does not contain the expected .model backbone."
        )

    backbone = outer_model.model

    if not hasattr(
        backbone,
        "neighbor_backend",
    ):
        raise AttributeError(
            f"Backbone {type(backbone).__name__} "
            "does not expose neighbor_backend."
        )

    backbone.neighbor_backend = "torch"

    if hasattr(
        backbone,
        "gather_query_chunk_size",
    ):
        backbone.gather_query_chunk_size = 4096

    print(
        f"Neighbor backend: "
        f"{backbone.neighbor_backend}"
    )

    if hasattr(
        backbone,
        "gather_query_chunk_size",
    ):
        print(
            "Gather query chunk size: "
            f"{backbone.gather_query_chunk_size}"
        )

    # -------------------------------------------------------------------------
    # 7. Load the graph Fourier basis
    # -------------------------------------------------------------------------
    graph_obj = torch.load(
        graph_basis_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(graph_obj, dict):
        raise TypeError(
            "Expected the graph-basis .pt file "
            "to contain a dictionary."
        )

    print(
        f"Graph basis keys: "
        f"{list(graph_obj.keys())}"
    )

    U = graph_obj.get("U")

    if U is None:
        U = graph_obj.get("eigenvectors")

    if U is None:
        U = graph_obj.get("evecs")

    eigenvalues = graph_obj.get(
        "eigenvalues"
    )

    if eigenvalues is None:
        eigenvalues = graph_obj.get(
            "evals"
        )

    if U is None:
        raise KeyError(
            "Graph basis does not contain "
            "U, eigenvectors, or evecs."
        )

    U = torch.as_tensor(
        U,
        dtype=torch.float32,
        device=device,
    )

    if U.ndim != 2:
        raise ValueError(
            "Expected U to have shape [N,K], "
            f"got {tuple(U.shape)}."
        )

    if U.shape[0] != dataset.num_points:
        raise ValueError(
            "Graph basis spatial size does not match the dataset: "
            f"U has N={U.shape[0]}, while the dataset has "
            f"N={dataset.num_points}."
        )

    # Prefer bands saved with the graph basis.
    saved_bands = graph_obj.get("bands")

    if saved_bands is not None:
        bands = {
            str(name): torch.as_tensor(
                indices,
                dtype=torch.long,
                device=device,
            )
            for name, indices
            in saved_bands.items()
        }
    else:
        if eigenvalues is None:
            raise KeyError(
                "Graph basis has neither "
                "saved bands nor eigenvalues."
            )

        eigenvalues = torch.as_tensor(
            eigenvalues,
            dtype=torch.float32,
            device=device,
        )

        generated_bands = (
            make_graph_frequency_bands(
                eigenvalues=eigenvalues,
                exclude_zero=True,
                split="thirds",
            )
        )

        bands = {
            str(name): torch.as_tensor(
                indices,
                dtype=torch.long,
                device=device,
            )
            for name, indices
            in generated_bands.items()
        }

    print(
        f"Graph basis shape: "
        f"{tuple(U.shape)}"
    )

    print(
        "Band sizes:",
        {
            name: int(indices.numel())
            for name, indices in bands.items()
        },
    )

    # -------------------------------------------------------------------------
    # 8. Select snapshots across the complete test split
    # -------------------------------------------------------------------------
    requested_snapshot_count = min(
        num_snapshots,
        len(dataset),
    )

    snapshot_indices = sorted(
        set(
            np.linspace(
                0,
                len(dataset) - 1,
                requested_snapshot_count,
            )
            .astype(int)
            .tolist()
        )
    )

    snapshot_count = len(
        snapshot_indices
    )

    if snapshot_count < 2:
        raise ValueError(
            "Cross-frequency band covariance "
            "requires at least two snapshots."
        )

    print(
        f"Selected {snapshot_count} snapshots "
        "distributed across the test split:"
    )
    print(snapshot_indices)

    # -------------------------------------------------------------------------
    # 9. Reconstruct the selected snapshots
    # -------------------------------------------------------------------------
    fields_true_list = []
    fields_pred_list = []

    with torch.no_grad():
        for position, snapshot_index in enumerate(
            snapshot_indices,
            start=1,
        ):
            print(
                f"Reconstructing snapshot "
                f"{position}/{snapshot_count} "
                f"(split index {snapshot_index})"
            )

            result = reconstruct_snapshot(
                model=model,
                dataset=dataset,
                device=device,
                snapshot_index=snapshot_index,
                cond_fields=cond_fields,
                n_obs_list=n_obs_list,
                n_steps=n_steps_generation,
                ode_solver=ode_solver,
            )

            # Each tensor has shape [1,N,C].
            fields_true_list.append(
                result["truth"]
                .detach()
                .cpu()
            )

            fields_pred_list.append(
                result["recon"]
                .detach()
                .cpu()
            )

    # Final tensors have shape [B,N,C].
    fields_true = torch.cat(
        fields_true_list,
        dim=0,
    ).to(device)

    fields_pred = torch.cat(
        fields_pred_list,
        dim=0,
    ).to(device)

    print(
        f"fields_true shape: "
        f"{tuple(fields_true.shape)}"
    )

    print(
        f"fields_pred shape: "
        f"{tuple(fields_pred.shape)}"
    )

    # -------------------------------------------------------------------------
    # 10. Optional conversion back to physical units
    # -------------------------------------------------------------------------
    if use_denorm:
        mean = dataset.mean.to(
            device=device,
            dtype=fields_true.dtype,
        ).view(1, 1, -1)

        std = dataset.std.to(
            device=device,
            dtype=fields_true.dtype,
        ).view(1, 1, -1)

        fields_true = (
            fields_true * std + mean
        )

        fields_pred = (
            fields_pred * std + mean
        )

        print(
            "Band energies will be computed "
            "in denormalized physical units."
        )
    else:
        print(
            "Band energies will be computed "
            "in normalized model space."
        )

    # -------------------------------------------------------------------------
    # 11. Compute same-frequency and cross-frequency band-energy metrics
    # -------------------------------------------------------------------------
    with torch.no_grad():
        metrics, payload = (
            _cross_spectral_coherence_band_metrics(
                fields_true=fields_true,
                fields_pred=fields_pred,
                U=U,
                bands=bands,
                field_pairs=None,
                eps=1e-12,
            )
        )

    # -------------------------------------------------------------------------
    # 12. Save visualization.py-style band-energy plots
    # -------------------------------------------------------------------------
    save_band_energy_diagnostic_plots(
        payload=payload,
        save_dir=output_dir,
        field_names=field_names,
        save_per_pair=True,
    )

    # -------------------------------------------------------------------------
    # 13. Save numerical metrics and payload
    # -------------------------------------------------------------------------
    metrics_path = (
        output_dir
        / "band_energy_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metrics,
            handle,
            indent=2,
        )

    payload_path = (
        output_dir
        / "band_energy_payload.npz"
    )

    np.savez_compressed(
        payload_path,
        **payload,
    )

    # -------------------------------------------------------------------------
    # 14. Save evaluation metadata
    # -------------------------------------------------------------------------
    metadata = {
        "run_dir": str(run_dir),
        "checkpoint": checkpoint_name,
        "checkpoint_path": str(
            checkpoint_path
        ),
        "graph_basis_path": str(
            graph_basis_path
        ),
        "data_path": str(data_path),
        "split": split,
        "num_snapshots": snapshot_count,
        "snapshot_indices": snapshot_indices,
        "cond_fields": list(
            cond_fields
        ),
        "n_obs_list": list(
            n_obs_list
        ),
        "n_steps_generation": (
            n_steps_generation
        ),
        "ode_solver": ode_solver,
        "use_denorm": use_denorm,
        "field_names": field_names,
        "output_dir": str(output_dir),
    }

    metadata_path = (
        output_dir
        / "band_energy_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # 15. Final summary
    # -------------------------------------------------------------------------
    print(
        "\nBand-energy evaluation complete."
    )

    print(
        f"Plots and numerical results saved to:\n"
        f"{output_dir}"
    )

    print("\nMetrics:")

    for key, value in metrics.items():
        print(
            f"  {key}: {value:.6e}"
        )

if __name__ == "__main__":
    main()