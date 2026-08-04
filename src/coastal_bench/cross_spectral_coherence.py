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

import argparse
import glob
import hashlib
import inspect
import json
import pickle
import sys
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# src/ on path so paper_style (figures) and evaluate_coherence (model mode) import.
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
for _d in (str(SCRIPT_DIR), str(SRC_DIR)):
    if _d not in sys.path:
        sys.path.append(_d)

import paper_style as ps

# ===========================================================================
# Graph-spectral basis & ensemble coherence estimation
# ===========================================================================
EPS = 1e-12

BAND_NAMES = ("large", "medium", "small")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def effective_coords(coords: np.ndarray, var_tol: float = 1e-9
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Drop coordinate axes with ~zero variance (e.g. a constant z-slice).

    Returns the reduced coordinates (N, d_eff) and the indices of the kept
    axes, so callers can reuse the same projection for plotting.
    """
    coords = np.asarray(coords, dtype=np.float64)
    var = coords.var(axis=0)
    keep = np.where(var > var_tol)[0]
    if keep.size == 0:                       # degenerate; keep first two axes
        keep = np.arange(min(2, coords.shape[1]))
    return coords[:, keep], keep


# ---------------------------------------------------------------------------
# Graph Laplacian
# ---------------------------------------------------------------------------
def build_knn_laplacian(
    coords: np.ndarray,
    k_neighbors: int = 12,
    sigma_scale: float = 1.0,
    weight: str = "gaussian",
) -> sp.csr_matrix:
    """Symmetric-normalized Laplacian of a Gaussian-weighted k-NN graph.

    L = I - D^{-1/2} W D^{-1/2},  eigenvalues in [0, 2].

    Parameters
    ----------
    coords      : (N, d) point coordinates (already reduced to effective dims).
    k_neighbors : neighbours per node (excluding self).
    sigma_scale : Gaussian length scale as a multiple of the median k-NN
                  distance. ``weight="binary"`` ignores this.
    """
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    n = coords.shape[0]
    k = int(min(k_neighbors, n - 1))
    tree = cKDTree(coords)
    # +1 because the first neighbour returned is the point itself.
    dist, idx = tree.query(coords, k=k + 1)
    dist, idx = dist[:, 1:], idx[:, 1:]              # drop self column

    if weight == "gaussian":
        sigma = sigma_scale * (np.median(dist) + EPS)
        w = np.exp(-(dist ** 2) / (2.0 * sigma ** 2))
    elif weight == "binary":
        w = np.ones_like(dist)
    else:
        raise ValueError(f"Unknown weight={weight!r}")

    rows = np.repeat(np.arange(n), k)
    cols = idx.reshape(-1)
    vals = w.reshape(-1)
    W = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    W = W.maximum(W.T)                                # symmetrize

    deg = np.asarray(W.sum(axis=1)).ravel()
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, EPS))
    Dis = sp.diags(d_inv_sqrt)
    L = sp.identity(n, format="csr") - Dis @ W @ Dis
    return (0.5 * (L + L.T)).tocsr()                 # enforce exact symmetry


# ---------------------------------------------------------------------------
# Spectral basis
# ---------------------------------------------------------------------------
@dataclass
class GraphSpectralBasis:
    eigvecs: np.ndarray          # (N, K) orthonormal; column 0 is the DC mode
    eigvals: np.ndarray          # (K,) ascending, eigvals[0] ~ 0
    freqs: np.ndarray            # (K,) graph frequency nu_k = sqrt(lambda_k)
    band_of_mode: np.ndarray     # (K,) int: 0/1/2 = large/medium/small, -1 = DC
    band_edges: np.ndarray       # (4,) frequency edges [f0, f1, f2, f3]
    keep_axes: np.ndarray        # indices of coordinate axes used for the graph
    meta: dict = field(default_factory=dict)

    @property
    def n_modes(self) -> int:
        return self.eigvecs.shape[1]

    def band_mode_mask(self, band: int) -> np.ndarray:
        return self.band_of_mode == band


def _assign_bands(freqs: np.ndarray, mode: str = "freq_equal"
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Split non-DC modes into 3 scale bands.

    ``freq_equal``  : equal thirds of the graph-frequency range (matches the
                      channel-flow figure's k_max/3 split).
    ``quantile``    : equal mode counts per band.
    """
    K = freqs.shape[0]
    band = np.full(K, -1, dtype=np.int64)
    nz = np.arange(1, K)                              # skip DC mode 0
    f = freqs[nz]
    if mode == "quantile":
        e = np.quantile(f, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    else:                                            # freq_equal
        e = np.linspace(f.min(), f.max(), 4)
    e[0], e[-1] = e[0] - EPS, e[-1] + EPS
    band[nz] = np.clip(np.digitize(f, e[1:-1]), 0, 2)
    return band, e


def _coords_hash(coords: np.ndarray, params: dict) -> str:
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(coords, dtype=np.float64).tobytes())
    h.update(repr(sorted(params.items())).encode())
    return h.hexdigest()[:16]


def compute_graph_basis(
    coords: np.ndarray,
    n_modes: int = 256,
    k_neighbors: int = 12,
    sigma_scale: float = 1.0,
    weight: str = "gaussian",
    band_mode: str = "freq_equal",
    cache_path: Optional[str | Path] = None,
    verbose: bool = True,
) -> GraphSpectralBasis:
    """Lowest ``n_modes`` graph-Laplacian eigenpairs (cached on disk).

    The DC (constant) mode is retained as column 0 but excluded from every band,
    so coherence and band-limited reconstructions describe fluctuations only.
    """
    coords_eff, keep = effective_coords(coords)
    n = coords_eff.shape[0]
    n_modes = int(min(n_modes, n - 1))

    params = dict(n_modes=n_modes, k_neighbors=k_neighbors,
                  sigma_scale=sigma_scale, weight=weight, band_mode=band_mode)
    key = _coords_hash(coords_eff, params)

    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            try:
                z = np.load(cache_path, allow_pickle=True)
                if str(z.get("key")) == key:
                    if verbose:
                        print(f"[graph_spectral] loaded cached basis: {cache_path}")
                    return GraphSpectralBasis(
                        eigvecs=z["eigvecs"], eigvals=z["eigvals"], freqs=z["freqs"],
                        band_of_mode=z["band_of_mode"], band_edges=z["band_edges"],
                        keep_axes=z["keep_axes"], meta=dict(z["meta"].item()),
                    )
                elif verbose:
                    print("[graph_spectral] cache key mismatch; recomputing basis.")
            except Exception as _cache_exc:
                if verbose:
                    print(f"[graph_spectral] corrupted cache ({_cache_exc}); "
                          "deleting and recomputing basis.")
                cache_path.unlink(missing_ok=True)

    if verbose:
        print(f"[graph_spectral] building k-NN Laplacian: N={n}, "
              f"k={k_neighbors}, modes={n_modes} ...")
    L = build_knn_laplacian(coords_eff, k_neighbors, sigma_scale, weight)

    if n <= max(4 * n_modes, 600):
        # Small graph (e.g. unit tests): a dense solve is faster and robust.
        vals, vecs = np.linalg.eigh(L.toarray())
        vals, vecs = vals[:n_modes], vecs[:, :n_modes]
    else:
        # Shift-invert just below 0 finds the smallest eigenvalues; the shift
        # makes (L - sigma I) SPD so the sparse factorization is stable.
        try:
            vals, vecs = eigsh(L, k=n_modes, sigma=-1e-3, which="LM")
        except Exception as exc:                     # pragma: no cover
            if verbose:
                print(f"[graph_spectral] shift-invert failed ({exc}); "
                      "falling back to which='SM'.")
            vals, vecs = eigsh(L, k=n_modes, which="SM", tol=1e-4)

    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    vals = np.clip(vals, 0.0, None)
    freqs = np.sqrt(vals)
    band_of_mode, band_edges = _assign_bands(freqs, band_mode)

    basis = GraphSpectralBasis(
        eigvecs=np.ascontiguousarray(vecs, dtype=np.float64),
        eigvals=vals, freqs=freqs, band_of_mode=band_of_mode,
        band_edges=band_edges, keep_axes=keep,
        meta=dict(n_points=n, **params),
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path, key=key, eigvecs=basis.eigvecs, eigvals=basis.eigvals,
            freqs=basis.freqs, band_of_mode=basis.band_of_mode,
            band_edges=basis.band_edges, keep_axes=basis.keep_axes,
            meta=basis.meta,
        )
        if verbose:
            print(f"[graph_spectral] cached basis -> {cache_path}")
    return basis


# ---------------------------------------------------------------------------
# Projection / reconstruction
# ---------------------------------------------------------------------------
def project_fields(basis: GraphSpectralBasis, fields: np.ndarray) -> np.ndarray:
    """Graph spectrum of every channel: a[k, c] = <phi_k, f_c>.

    fields : (N, C)  ->  coeffs (K, C).
    """
    fields = np.asarray(fields, dtype=np.float64)
    return basis.eigvecs.T @ fields


def band_limited_field(basis: GraphSpectralBasis, fields: np.ndarray,
                       band: int) -> np.ndarray:
    """Spatial reconstruction from the modes of one scale band.

    f_B(x) = sum_{k in band} a_k phi_k(x).  Returns (N, C); always zero-mean
    because the DC mode is excluded from every band.
    """
    coeffs = project_fields(basis, fields)           # (K, C)
    mask = basis.band_mode_mask(band)
    return basis.eigvecs[:, mask] @ coeffs[mask, :]


def field_pairs(n_fields: int) -> List[Tuple[int, int]]:
    return list(combinations(range(n_fields), 2))


# ---------------------------------------------------------------------------
# Ensemble coherence
# ---------------------------------------------------------------------------
@dataclass
class SpectralEstimate:
    """Welch-style ensemble auto/cross graph spectra (averaged over snapshots).

    auto  : (K, C)              P_c[k]   = <|a_k^c|^2>
    cross : (K, n_pairs)        P_ij[k]  = <a_k^i a_k^j>   (signed, real basis)
    pairs : list of (i, j)
    n_snapshots : ensemble size used for the average.
    """
    auto: np.ndarray
    cross: np.ndarray
    pairs: List[Tuple[int, int]]
    n_snapshots: int


def accumulate_spectra(basis: GraphSpectralBasis,
                       fields_stack: Sequence[np.ndarray]) -> SpectralEstimate:
    """Average auto/cross graph spectra over an ensemble of snapshots.

    fields_stack : sequence of (N, C) field arrays (one per snapshot).
    """
    coeffs = np.stack([project_fields(basis, f) for f in fields_stack], axis=0)
    M, K, C = coeffs.shape
    auto = np.mean(coeffs ** 2, axis=0)              # (K, C)
    pairs = field_pairs(C)
    cross = np.empty((K, len(pairs)), dtype=np.float64)
    for p, (i, j) in enumerate(pairs):
        cross[:, p] = np.mean(coeffs[:, :, i] * coeffs[:, :, j], axis=0)
    return SpectralEstimate(auto=auto, cross=cross, pairs=pairs, n_snapshots=M)


def coherence_from_bins(est: SpectralEstimate, bin_assignment: np.ndarray
                        ) -> np.ndarray:
    """Magnitude-squared coherence per (bin, pair).

    Aggregates the auto/cross spectra within each bin (frequency smoothing),
    then forms gamma^2_{ij}[B] = |sum_B P_ij|^2 / (sum_B P_i  sum_B P_j).

    bin_assignment : (K,) int bin id per mode; negative ids are dropped
                     (used to exclude the DC mode).
    Returns gamma2 of shape (n_bins, n_pairs), values in [0, 1].
    """
    bin_ids = np.asarray(bin_assignment)
    valid = np.unique(bin_ids[bin_ids >= 0])
    n_bins = int(valid.max()) + 1 if valid.size else 0
    gamma2 = np.zeros((n_bins, len(est.pairs)), dtype=np.float64)
    for b in valid:
        m = bin_ids == b
        a = est.auto[m].sum(axis=0)                  # (C,)
        x = est.cross[m].sum(axis=0)                 # (n_pairs,)
        for p, (i, j) in enumerate(est.pairs):
            gamma2[b, p] = x[p] ** 2 / (a[i] * a[j] + EPS)
    return np.clip(gamma2, 0.0, 1.0)


def band_coherence(est: SpectralEstimate, basis: GraphSpectralBasis
                   ) -> np.ndarray:
    """gamma^2 per (scale-band, pair); shape (3, n_pairs)."""
    return coherence_from_bins(est, basis.band_of_mode)


def coherence_matrix(gamma2_band: np.ndarray, pairs: List[Tuple[int, int]],
                     n_fields: int) -> np.ndarray:
    """Pack per-pair band coherence into symmetric (n_bands, C, C) matrices.

    Diagonal is set to 1 (a field is perfectly coherent with itself).
    """
    n_bands = gamma2_band.shape[0]
    mat = np.zeros((n_bands, n_fields, n_fields), dtype=np.float64)
    for b in range(n_bands):
        for d in range(n_fields):
            mat[b, d, d] = 1.0
        for p, (i, j) in enumerate(pairs):
            mat[b, i, j] = mat[b, j, i] = gamma2_band[b, p]
    return mat


def frequency_bins(basis: GraphSpectralBasis, n_bins: int = 24
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Contiguous equal-width frequency bins over non-DC modes (for curves).

    Returns (bin_assignment (K,), bin_centers (n_bins,)) in graph-frequency
    units; the DC mode gets bin id -1.
    """
    f = basis.freqs
    nz = np.arange(1, basis.n_modes)
    edges = np.linspace(f[nz].min(), f[nz].max(), n_bins + 1)
    edges[0] -= EPS
    bin_assignment = np.full(basis.n_modes, -1, dtype=np.int64)
    bin_assignment[nz] = np.clip(np.digitize(f[nz], edges[1:-1]), 0, n_bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return bin_assignment, centers


# ---------------------------------------------------------------------------
# PDF loss decomposition (per field-pair, per band)
# ---------------------------------------------------------------------------
def coherence_loss_breakdown(
    est_pred: SpectralEstimate,
    est_data: SpectralEstimate,
    basis: GraphSpectralBasis,
    n_freq_bins: int = 24,
) -> Dict[str, np.ndarray]:
    """L_coh decomposed by field pair and scale band (PDF sections 4-5).

    Uses the binned per-frequency coherence so the per-mode degeneracy is
    removed: L^B_{c1c2} = sum_{k in B} (gamma^2_pred[k] - gamma^2_data[k])^2.

    Returns dict with:
        per_pair_band : (n_bands, n_pairs)   raw band losses L^B_{c1c2}
        per_pair      : (n_pairs,)           total L_{c1c2}
        per_band      : (n_bands,)           total L^B
        total         : scalar               L_coh
        band_fraction : (n_bands,)           r_B in [0, 1]
    """
    binned, _ = frequency_bins(basis, n_freq_bins)
    g_pred = coherence_from_bins(est_pred, binned)   # (n_freq_bins, n_pairs)
    g_data = coherence_from_bins(est_data, binned)
    diff2 = (g_pred - g_data) ** 2                    # (n_freq_bins, n_pairs)

    # Map each frequency bin to a scale band via its mean mode frequency.
    _, centers = frequency_bins(basis, n_freq_bins)
    edges = basis.band_edges
    bin_band = np.clip(np.digitize(centers, edges[1:-1]), 0, 2)

    n_bands = 3
    per_pair_band = np.zeros((n_bands, diff2.shape[1]))
    for b in range(n_bands):
        per_pair_band[b] = diff2[bin_band == b].sum(axis=0)

    per_pair = diff2.sum(axis=0)
    per_band = per_pair_band.sum(axis=1)
    total = float(per_pair.sum())
    return dict(
        per_pair_band=per_pair_band, per_pair=per_pair, per_band=per_band,
        total=total, band_fraction=per_band / (total + 1e-9),
    )


def rank_pairs_by_coherence(est: SpectralEstimate, basis: GraphSpectralBasis
                            ) -> np.ndarray:
    """Pair indices sorted by descending mean band coherence (strongest first)."""
    gb = band_coherence(est, basis)                  # (3, n_pairs)
    return np.argsort(-gb.mean(axis=0))


# ---------------------------------------------------------------------------
# Band energy (PDF section 6)
# ---------------------------------------------------------------------------
@dataclass
class BandEnergyEstimate:
    """Per-sample band energies, parallel to SpectralEstimate.

    raw      : (B, n_bands, C)  E_{b,c}^{B_m} = sum_{k in B_m} |X̂_{b,k,c}|^2
    centered : (B, n_bands, C)  Ẽ_{b,c}^{B_m} = E_{b,c}^{B_m} - mean_b(E)
    n_snapshots : ensemble size B
    """
    raw: np.ndarray          # (B, n_bands, C)
    centered: np.ndarray     # (B, n_bands, C)
    n_snapshots: int


def compute_band_energies(basis: GraphSpectralBasis,
                          fields_stack: Sequence[np.ndarray]) -> BandEnergyEstimate:
    """Band energy E_{b,c}^{B_m} and centered band energy Ẽ_{b,c}^{B_m}.

    For each snapshot b, field c, and scale band B_m:
        E_{b,c}^{B_m} = sum_{k in B_m} |X̂_{b,k,c}|^2
        Ẽ_{b,c}^{B_m} = E_{b,c}^{B_m} - (1/B) sum_s E_{s,c}^{B_m}

    fields_stack : sequence of (N, C) arrays, one per snapshot.
    Returns BandEnergyEstimate; raw and centered shapes are (B, n_bands, C).
    """
    coeffs = np.stack([project_fields(basis, f) for f in fields_stack], axis=0)
    B, K, C = coeffs.shape
    n_bands = 3                                       # large / medium / small
    raw = np.zeros((B, n_bands, C), dtype=np.float64)
    for m in range(n_bands):
        mask = basis.band_mode_mask(m)               # (K,) bool
        raw[:, m, :] = (coeffs[:, mask, :] ** 2).sum(axis=1)   # (B, C)
    centered = raw - raw.mean(axis=0, keepdims=True) # (B, n_bands, C)
    return BandEnergyEstimate(raw=raw, centered=centered, n_snapshots=B)


# ---------------------------------------------------------------------------
# Normalized cross-band coupling (PDF section 7)
# ---------------------------------------------------------------------------
def cross_band_coupling_scores(be: BandEnergyEstimate,
                               pairs: List[Tuple[int, int]]) -> np.ndarray:
    """Normalized cross-band coupling matrix Q̂_{c1c2}^{Bi,Bj}.

    For each field pair (c1, c2) and band pair (Bi, Bj):
        Ŝ_{c1c2}^{Bi,Bj} = (1/B) sum_b Ẽ_{b,c1}^{Bi} Ẽ_{b,c2}^{Bj}
        q̂_{c1c2}^{Bi,Bj} = (Ŝ_{c1c2}^{Bi,Bj})^2 /
                             (Ŝ_{c1c1}^{Bi,Bi} Ŝ_{c2c2}^{Bj,Bj} + eps)

    Diagonal entries (i == j) capture same-band coupling at the band-energy
    level; off-diagonal entries capture cross-frequency or cross-scale coupling.

    be    : BandEnergyEstimate from compute_band_energies.
    pairs : list of (c1, c2) with c1 < c2, e.g. from field_pairs(C).
    Returns Q of shape (n_pairs, n_bands, n_bands), values in [0, 1].
    """
    E = be.centered                                  # (B, n_bands, C)
    B, n_bands, C = E.shape

    # Full cross-covariance tensor S[Bi, c1, Bj, c2] = (1/B) sum_b E[b,Bi,c1]*E[b,Bj,c2]
    S_all = np.einsum('bic,bjd->icjd', E, E) / B    # (n_bands, C, n_bands, C)

    # Self-covariance: S_{cc}^{Bm,Bm} = (1/B) sum_b Ẽ[b,m,c]^2
    self_cov = np.einsum('bic,bic->ic', E, E) / B   # (n_bands, C)

    Q = np.zeros((len(pairs), n_bands, n_bands), dtype=np.float64)
    for p, (c1, c2) in enumerate(pairs):
        S_cross = S_all[:, c1, :, c2]               # (n_bands, n_bands)
        sc1 = self_cov[:, c1, np.newaxis]            # (n_bands, 1)  Ŝ_{c1c1}^{Bi,Bi}
        sc2 = self_cov[np.newaxis, :, c2]            # (1, n_bands)  Ŝ_{c2c2}^{Bj,Bj}
        Q[p] = S_cross ** 2 / (sc1 * sc2 + EPS)
    return np.clip(Q, 0.0, None)


def compute_S_matrix_from_be(be: BandEnergyEstimate) -> np.ndarray:
    """Full cross-band covariance tensor shaped (3, 3, C, C).

    S[Bi, Bj, c1, c2] = (1/B) sum_b Ẽ_{b,c1}^{Bi} * Ẽ_{b,c2}^{Bj}

    This is the un-normalised precursor to Q; needed for the cross-band energy
    ratio panels (|S_pred| / |S_GT|).
    """
    E = be.centered                                  # (B, n_bands, C)
    B = E.shape[0]
    # einsum 'bic,bjd->icjd' -> (n_bands, C, n_bands, C); transpose to (3,3,C,C)
    S_icjd = np.einsum('bic,bjd->icjd', E, E) / B
    n_bands, C = S_icjd.shape[0], S_icjd.shape[1]
    return S_icjd.transpose(0, 2, 1, 3)             # (n_bands, n_bands, C, C)


# ---------------------------------------------------------------------------
# Cross-frequency coupling loss (PDF section 8)
# ---------------------------------------------------------------------------
def cross_frequency_coupling_loss(Q_pred: np.ndarray,
                                  Q_data: np.ndarray) -> Dict[str, np.ndarray]:
    """L_crossfreq: MSE on off-diagonal cross-band coupling scores (PDF section 8).

    Only off-diagonal band-pair entries (i != j) contribute, capturing
    cross-scale coupling while excluding same-band terms already measured by
    the same-frequency coherence loss.

        L_crossfreq = sum_{c1<c2} sum_{i!=j} (q̂^pred - q̂^data)^2

    Q_pred : (n_pairs, n_bands, n_bands)  predicted coupling scores.
    Q_data : (n_pairs, n_bands, n_bands)  target coupling scores.
    Returns dict with:
        per_pair      : (n_pairs,)           L_crossfreq per field pair
        per_band_pair : (n_bands, n_bands)   summed over field pairs (off-diag only)
        total         : float                L_crossfreq
    """
    diff2 = (Q_pred - Q_data) ** 2                  # (n_pairs, n_bands, n_bands)
    n_bands = Q_pred.shape[1]
    offdiag = ~np.eye(n_bands, dtype=bool)           # (n_bands, n_bands) True where i!=j
    per_pair = (diff2 * offdiag[np.newaxis]).sum(axis=(1, 2))   # (n_pairs,)
    per_band_pair = diff2.sum(axis=0) * offdiag      # (n_bands, n_bands)
    total = float(per_pair.sum())
    return dict(per_pair=per_pair, per_band_pair=per_band_pair, total=total)


# ---------------------------------------------------------------------------
# Physical coherence objective (PDF section 9)
# ---------------------------------------------------------------------------
def physical_coherence_loss(loss_same: float, loss_crossfreq: float,
                             eta_crossfreq: float = 1.0) -> float:
    """L_phys-coh = L_same + eta_crossfreq * L_crossfreq  (PDF eq. section 9).

    loss_same      : scalar total from coherence_loss_breakdown(...)['total'].
    loss_crossfreq : scalar total from cross_frequency_coupling_loss(...)['total'].
    eta_crossfreq  : weight >= 0 controlling cross-frequency coupling strength.
    """
    return loss_same + eta_crossfreq * loss_crossfreq


def total_training_loss(L_gen: float, L_phys_coh: float,
                        lambda_phys: float = 1.0) -> float:
    """L_total = L_gen + lambda_phys * L_phys-coh  (PDF eq. section 9).

    L_gen        : generative (reconstruction) loss.
    L_phys_coh   : output of physical_coherence_loss.
    lambda_phys  : weight >= 0 scaling the physical coherence regularizer.
    """
    return L_gen + lambda_phys * L_phys_coh


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
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=val, cmap=cmap, vmin=vmin, vmax=vmax,
                    s=s, marker="o", linewidths=0.0, rasterized=True)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
    return sc


def _sym(vals, pct):
    a = float(np.percentile(np.abs(vals[np.isfinite(vals)]), pct)) or 1.0
    return -a, a


# ---------------------------------------------------------------------------
# Block A : spatial co-fluctuation
# ---------------------------------------------------------------------------
def draw_spatial_block(
    sub,
    xy: np.ndarray,
    basis: GraphSpectralBasis,
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
    band_names = ("Large", "Medium", "Small")
    col_titles = [f"{names[i]}  (truth)", f"{names[j]}  (truth)",
                  f"{names[i]}·{names[j]}  (truth)",
                  f"{names[i]}·{names[j]}  (FFM)"]

    gs = sub.add_gridspec(3, 4, hspace=0.12, wspace=0.05,
                          left=0.07, right=0.92, top=0.88, bottom=0.06)

    for b in range(3):
        f_gt = band_limited_field(basis, fields_gt, b)       # (N, C)
        f_ff = band_limited_field(basis, fields_ffm, b)
        f1, f2 = f_gt[:, i], f_gt[:, j]

        # Co-fluctuation: normalize each band-limited field to unit std, multiply.
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

        ax0.set_ylabel(band_names[b], fontsize=9.5, fontweight="bold",
                       rotation=90, labelpad=6)

        # gamma^2 annotations under the co-fluctuation maps.
        ax2.text(0.5, -0.04, f"$\\gamma^2$={gamma_gt[b]:.2f}", color=COL_GT,
                 transform=ax2.transAxes, ha="center", va="top", fontsize=8.0)
        ax3.text(0.5, -0.04, f"$\\gamma^2$={gamma_ffm[b]:.2f}", color=COL_FFM,
                 transform=ax3.transAxes, ha="center", va="top", fontsize=8.0,
                 fontweight="bold")

        if b == 0:
            for ax, t in zip((ax0, ax1, ax2, ax3), col_titles):
                ax.set_title(t, fontsize=8.5, pad=4)

    # Two shared colorbars: fluctuation (qualitative) and co-fluctuation.
    cax1 = sub.add_axes([0.935, 0.50, 0.012, 0.34])
    cb1 = sub.figure.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(-1, 1), cmap=CMAP_FLUCT),
        cax=cax1)
    cb1.set_ticks([-1, 0, 1]); cb1.set_ticklabels(["–", "0", "+"])
    cb1.set_label("band fluctuation\n(per-panel scaled)", fontsize=7.0)
    cb1.ax.tick_params(labelsize=7.0)

    cax2 = sub.add_axes([0.935, 0.08, 0.012, 0.34])
    cb2 = sub.figure.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(-1, 1), cmap=CMAP_COFLUCT),
        cax=cax2)
    cb2.set_ticks([-1, 0, 1]); cb2.set_ticklabels(["oppose", "0", "align"])
    cb2.set_label("co-fluctuation", fontsize=7.0)
    cb2.ax.tick_params(labelsize=7.0)

    if title:
        sub.suptitle(title, fontsize=10.5, fontweight="bold", y=0.98)


# ---------------------------------------------------------------------------
# Block B : all-pair scale matrices
# ---------------------------------------------------------------------------
def draw_matrix_block(
    sub,
    mat_gt: np.ndarray,         # (3, C, C) ground-truth coherence
    mat_err: np.ndarray,        # (3, C, C) FFM - GT
    names: Sequence[str],
    title: Optional[str] = None,
) -> None:
    band_names = ("large", "medium", "small")
    C = mat_gt.shape[1]
    err_lim = float(np.percentile(np.abs(mat_err), 99)) or 1.0

    gs = sub.add_gridspec(2, 3, hspace=0.38, wspace=0.18,
                          left=0.10, right=0.90, top=0.86, bottom=0.10)

    def _draw_one(ax, M, kind, vmin, vmax, show_y):
        cmap = CMAP_COH if kind == "coh" else ps.CMAP_ERROR
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xticks(range(C)); ax.set_yticks(range(C))
        ax.set_xticklabels(names, fontsize=7.0, rotation=45, ha="right")
        ax.set_yticklabels(names if show_y else [], fontsize=7.0)
        for a in range(C):
            for b in range(C):
                v = M[a, b]
                if kind == "coh":
                    txt = f"{v:.2f}"
                    light = v < 0.55              # magma dark at low gamma^2
                else:
                    txt = f"{v:+.2f}"
                    light = abs(v) > 0.55 * vmax  # RdBu_r dark at extremes
                ax.text(b, a, txt, ha="center", va="center", fontsize=5.6,
                        color="white" if light else "black")
        return im

    im_gt = None
    for b in range(3):
        ax = sub.add_subplot(gs[0, b])
        im_gt = _draw_one(ax, mat_gt[b], "coh", 0.0, 1.0, show_y=(b == 0))
        ax.set_title(f"GT — {band_names[b]}", fontsize=8.5)
    im_err = None
    for b in range(3):
        ax = sub.add_subplot(gs[1, b])
        im_err = _draw_one(ax, mat_err[b], "err", -err_lim, err_lim,
                           show_y=(b == 0))
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
# Block C : coherence curves + per-pair loss bars
# ---------------------------------------------------------------------------
def _shade_bands(ax, band_edges):
    for b in range(3):
        ax.axvspan(band_edges[b], band_edges[b + 1], color=BAND_TINT[b],
                   alpha=0.7, lw=0)


def draw_curve_block(
    sub,
    basis: GraphSpectralBasis,
    est_gt: SpectralEstimate,
    est_ffm: SpectralEstimate,
    pair_idx: int,
    pair: Tuple[int, int],
    names: Sequence[str],
    loss: Dict[str, np.ndarray],
    cfg: VizConfig,
    title: Optional[str] = None,
) -> None:
    bin_assign, centers = frequency_bins(basis, cfg.n_freq_bins)
    g_gt = coherence_from_bins(est_gt, bin_assign)        # (n_bins, n_pairs)
    g_ff = coherence_from_bins(est_ffm, bin_assign)
    gb_gt = band_coherence(est_gt, basis)
    gb_ff = band_coherence(est_ffm, basis)
    band_centers = [0.5 * (basis.band_edges[b] + basis.band_edges[b + 1])
                    for b in range(3)]

    gs = sub.add_gridspec(1, 2, wspace=0.28, width_ratios=[1.05, 1.25],
                          left=0.08, right=0.97, top=0.84, bottom=0.18)

    # -- left: selected-pair coherence spectrum --
    axc = sub.add_subplot(gs[0, 0])
    _shade_bands(axc, basis.band_edges)
    axc.plot(centers, g_gt[:, pair_idx], color=COL_GT, lw=1.6, label="Ground truth")
    axc.plot(centers, g_ff[:, pair_idx], color=COL_FFM, lw=1.6, ls="--", label="FFM")
    axc.scatter(band_centers, gb_gt[:, pair_idx], color=COL_GT, s=34, zorder=5,
                edgecolors="white", linewidths=0.6)
    axc.scatter(band_centers, gb_ff[:, pair_idx], color=COL_FFM, s=34, zorder=5,
                marker="D", edgecolors="white", linewidths=0.6)
    axc.set_xlabel("graph frequency  $\\nu_k=\\sqrt{\\lambda_k}$", fontsize=8.5)
    axc.set_ylabel("$\\gamma^2$ coherence", fontsize=8.5)
    axc.set_ylim(0, 1.02)
    axc.set_xlim(centers.min(), centers.max())
    axc.set_title(f"Coherence spectrum: {pair_label(names, *pair)}", fontsize=9)
    axc.legend(fontsize=7.5, loc="upper right", framealpha=0.85)
    for b, nm in enumerate(("large", "medium", "small")):
        axc.text(band_centers[b], 0.04, nm, ha="center", fontsize=7.0,
                 color=BAND_EDGE_C[b], style="italic")

    # -- right: per-field-pair coherence loss, stacked by band --
    axb = sub.add_subplot(gs[0, 1])
    ppb = loss["per_pair_band"]                           # (3, n_pairs)
    plabels = [pair_label(names, a, b) for (a, b) in est_gt.pairs]
    order = np.argsort(-loss["per_pair"])                 # worst pairs first
    x = np.arange(len(plabels))
    bottom = np.zeros(len(plabels))
    for b in range(3):
        axb.bar(x, ppb[b][order], bottom=bottom, width=0.74,
                color=BAND_TINT[b], edgecolor=BAND_EDGE_C[b], lw=0.6,
                label=("large", "medium", "small")[b])
        bottom += ppb[b][order]
    axb.set_xticks(x)
    axb.set_xticklabels([plabels[k] for k in order], rotation=55, ha="right",
                        fontsize=6.6)
    axb.set_ylabel("$L_{\\mathrm{coh}}^{c_1 c_2}$  (FFM vs GT)", fontsize=8.5)
    axb.set_title("Coherence-loss decomposition by field pair", fontsize=9)
    axb.legend(fontsize=7.0, title="scale band", title_fontsize=7.0,
               loc="upper right")
    axb.margins(x=0.01)

    if title:
        sub.suptitle(title, fontsize=10.5, fontweight="bold", y=0.99)


# ---------------------------------------------------------------------------
# Optional standalone: all-pairs coherence small multiples
# ---------------------------------------------------------------------------
def figure_all_pair_curves(
    basis: GraphSpectralBasis,
    est_gt: SpectralEstimate,
    est_ffm: SpectralEstimate,
    names: Sequence[str],
    cfg: VizConfig,
) -> plt.Figure:
    bin_assign, centers = frequency_bins(basis, cfg.n_freq_bins)
    g_gt = coherence_from_bins(est_gt, bin_assign)
    g_ff = coherence_from_bins(est_ffm, bin_assign)
    pairs = est_gt.pairs
    ncol = 5
    nrow = int(np.ceil(len(pairs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.1 * ncol, 1.7 * nrow),
                             squeeze=False)
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
    fig.suptitle("Cross-spectral coherence per field pair (— GT,  -- FFM)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig

################################################################################
# ---------------------------------------------------------------------------
# Block D : cross-band coupling matrices
# ---------------------------------------------------------------------------
def draw_crossband_matrix_block(
    sub,
    Q_gt: np.ndarray,           # (n_pairs, 3, 3)
    Q_ffm: np.ndarray,          # (n_pairs, 3, 3)
    pairs: List[Tuple[int, int]],
    names: Sequence[str],
    n_show: int = 6,
    title: Optional[str] = None,
) -> None:
    """GT and error (FFM-GT) Q̂_{c1c2}^{Bi,Bj} matrices for the top field pairs.

    Pairs are ranked by descending mean off-diagonal GT coupling.  Layout
    mirrors Block B: GT row on top, signed-error row on bottom, one column
    per field pair.  Diagonal cells (excluded from L_crossfreq) are marked "*".
    """
    n_bands = Q_gt.shape[1]
    band_tick = [b[0].upper() for b in BAND_NAMES]   # ["L", "M", "S"]
    offdiag = ~np.eye(n_bands, dtype=bool)

    mean_od = np.array([(Q_gt[p] * offdiag).mean() for p in range(len(pairs))])
    order = np.argsort(-mean_od)
    n_show = min(n_show, len(pairs))
    show = order[:n_show]

    Q_err = Q_ffm - Q_gt
    err_lim = max(float(np.percentile(np.abs(Q_err[show]), 99)), 0.05)

    gs = sub.add_gridspec(2, n_show, hspace=0.50, wspace=0.35,
                          left=0.08, right=0.87, top=0.86, bottom=0.08)

    im_gt = im_err = None
    for col, p in enumerate(show):
        c1, c2 = pairs[p]
        lab = pair_label(names, c1, c2)
        ax_top = sub.add_subplot(gs[0, col])
        ax_bot = sub.add_subplot(gs[1, col])

        for ax, M, kind, vmin, vmax in (
            (ax_top, Q_gt[p],  "coh",  0.0,      1.0),
            (ax_bot, Q_err[p], "err", -err_lim, err_lim),
        ):
            cmap = CMAP_COH if kind == "coh" else ps.CMAP_ERROR
            im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal",
                           interpolation="nearest")
            ax.set_xticks(range(n_bands))
            ax.set_yticks(range(n_bands))
            ax.set_xticklabels(band_tick, fontsize=6.5)
            ax.set_yticklabels(band_tick if col == 0 else [], fontsize=6.5)
            for bi in range(n_bands):
                for bj in range(n_bands):
                    v = M[bi, bj]
                    if bi == bj:
                        ax.text(bj, bi, "*", ha="center", va="center",
                                fontsize=7.5, color="gray", zorder=3)
                    else:
                        txt = f"{v:.2f}" if kind == "coh" else f"{v:+.2f}"
                        light = (v < 0.4 if kind == "coh"
                                 else abs(v) > 0.5 * err_lim)
                        ax.text(bj, bi, txt, ha="center", va="center",
                                fontsize=5.5,
                                color="white" if light else "black", zorder=3)
            if kind == "coh":
                im_gt = im
                ax.set_title(lab, fontsize=7.5, pad=3)
            else:
                im_err = im
        if col == 0:
            ax_top.set_ylabel("GT  ($B_{c_1}$)", fontsize=7.0)
            ax_bot.set_ylabel("FFM\u2013GT  ($B_{c_1}$)", fontsize=7.0)
        ax_bot.set_xlabel("$B_{c_2}$", fontsize=7.0)

    if im_gt is not None:
        cax1 = sub.add_axes([0.893, 0.52, 0.013, 0.30])
        cb1 = sub.figure.colorbar(im_gt, cax=cax1)
        cb1.set_label("$\\hat{q}$ (GT)", fontsize=7.5)
        cb1.ax.tick_params(labelsize=7.0)
    if im_err is not None:
        cax2 = sub.add_axes([0.893, 0.10, 0.013, 0.30])
        cb2 = sub.figure.colorbar(im_err, cax=cax2)
        cb2.set_label("$\\Delta\\hat{q}$ (FFM\u2013GT)", fontsize=7.5)
        cb2.ax.tick_params(labelsize=7.0)

    if title:
        sub.suptitle(title, fontsize=10.5, fontweight="bold", y=0.97)


# ---------------------------------------------------------------------------
# Block E : band-energy scatter plots
# ---------------------------------------------------------------------------
def draw_band_energy_scatter(
    sub,
    be_gt: BandEnergyEstimate,
    be_ffm: BandEnergyEstimate,
    pair: Tuple[int, int],
    names: Sequence[str],
    Q_gt_pair: np.ndarray,      # (3, 3) cross-band coupling for this pair, GT
    Q_ffm_pair: np.ndarray,     # (3, 3) cross-band coupling for this pair, FFM
    title: Optional[str] = None,
) -> None:
    """3x3 scatter of centered band energies for one field pair.

    Row i / column j shows the joint distribution of Ẽ_{c1}^{Bi} (y) vs
    Ẽ_{c2}^{Bj} (x) across the batch.  GT is shown in black, the FFM
    reconstruction in orange.  Each panel is annotated with q̂ (GT and FFM).
    Diagonal panels (Bi=Bj) receive the corresponding band background tint.
    """
    c1, c2 = pair
    E_gt  = be_gt.centered                               # (B, 3, C)
    E_ffm = be_ffm.centered
    B = E_gt.shape[0]
    n_bands = 3
    s_pt = max(2.5, 150.0 / B)
    band_init = [b[0].upper() for b in BAND_NAMES]       # ["L", "M", "S"]

    gs = sub.add_gridspec(n_bands, n_bands, hspace=0.60, wspace=0.50,
                          left=0.10, right=0.97, top=0.88, bottom=0.10)

    sc_gt = sc_ffm = first_ax = None
    for bi in range(n_bands):
        for bj in range(n_bands):
            ax = sub.add_subplot(gs[bi, bj])
            if first_ax is None:
                first_ax = ax

            x_gt  = E_gt[:,  bj, c2]
            y_gt  = E_gt[:,  bi, c1]
            x_ffm = E_ffm[:, bj, c2]
            y_ffm = E_ffm[:, bi, c1]

            if bi == bj:
                ax.set_facecolor(BAND_TINT[bi])
            _sg  = ax.scatter(x_gt,  y_gt,  s=s_pt, color=COL_GT,  alpha=0.65,
                               linewidths=0.0, rasterized=True)
            _sff = ax.scatter(x_ffm, y_ffm, s=s_pt, color=COL_FFM, alpha=0.65,
                               linewidths=0.0, rasterized=True)
            if sc_gt is None:
                sc_gt, sc_ffm = _sg, _sff

            ax.axhline(0, color="gray", lw=0.5, ls=":")
            ax.axvline(0, color="gray", lw=0.5, ls=":")
            ax.tick_params(labelsize=5.5)

            q_gt_v  = Q_gt_pair[bi, bj]
            q_ffm_v = Q_ffm_pair[bi, bj]
            ax.text(0.04, 0.97,
                    f"$\\hat{{q}}_{{\\mathrm{{GT}}}}$={q_gt_v:.2f}",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=5.5, color=COL_GT)
            ax.text(0.97, 0.03,
                    f"$\\hat{{q}}_{{\\mathrm{{FFM}}}}$={q_ffm_v:.2f}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=5.5, color=COL_FFM)

            if bj == 0:
                ax.set_ylabel(
                    f"$\\tilde{{E}}_{{{names[c1]}}}^{{{band_init[bi]}}}$",
                    fontsize=6.5)
            if bi == n_bands - 1:
                ax.set_xlabel(
                    f"$\\tilde{{E}}_{{{names[c2]}}}^{{{band_init[bj]}}}$",
                    fontsize=6.5)

    if first_ax is not None and sc_gt is not None:
        first_ax.legend([sc_gt, sc_ffm], ["GT", "FFM"], fontsize=6.5,
                        loc="lower right", markerscale=1.5, framealpha=0.75)
    if title:
        sub.suptitle(title, fontsize=10.5, fontweight="bold", y=0.99)


# ---------------------------------------------------------------------------
# Block F : per-field auto-spectrum curves
# ---------------------------------------------------------------------------
def draw_auto_spectrum_block(
    sub,
    est_gt: SpectralEstimate,
    est_ffm: SpectralEstimate,
    basis: GraphSpectralBasis,
    names: Sequence[str],
    cfg: VizConfig,
    title: Optional[str] = None,
) -> None:
    """Per-field binned auto-spectrum p̂_c[k] vs graph frequency (log scale).

    Shows GT and FFM auto-spectra for each physical field as small-multiple
    line plots, using the same frequency binning as the coherence-curve plots.
    Band shading matches the existing coherence-spectrum style.
    """
    bin_assign, centers = frequency_bins(basis, cfg.n_freq_bins)
    C = est_gt.auto.shape[1]
    n_bins = centers.shape[0]

    auto_gt = np.zeros((n_bins, C))
    auto_ff = np.zeros((n_bins, C))
    for b in range(n_bins):
        m = bin_assign == b
        if m.any():
            auto_gt[b] = est_gt.auto[m].sum(axis=0)    # (C,)
            auto_ff[b] = est_ffm.auto[m].sum(axis=0)
    auto_gt = np.maximum(auto_gt, EPS)
    auto_ff = np.maximum(auto_ff, EPS)

    ncol = min(4, C)
    nrow = int(np.ceil(C / ncol))
    gs = sub.add_gridspec(nrow, ncol, hspace=0.55, wspace=0.38,
                          left=0.08, right=0.97, top=0.88, bottom=0.10)

    for c in range(C):
        row, col = c // ncol, c % ncol
        ax = sub.add_subplot(gs[row, col])
        _shade_bands(ax, basis.band_edges)
        ax.plot(centers, auto_gt[:, c], color=COL_GT,  lw=1.4,
                label="GT"  if c == 0 else None)
        ax.plot(centers, auto_ff[:, c], color=COL_FFM, lw=1.4, ls="--",
                label="FFM" if c == 0 else None)
        ax.set_yscale("log")
        ax.set_title(names[c], fontsize=8.5)
        ax.set_xlim(centers.min(), centers.max())
        ax.tick_params(labelsize=6.5)
        if row == nrow - 1:
            ax.set_xlabel("$\\nu_k=\\sqrt{\\lambda_k}$", fontsize=8.0)
        if col == 0:
            ax.set_ylabel("$\\hat{p}_c$  (spectral energy)", fontsize=7.5)
        if c == 0:
            ax.legend(fontsize=7.0, loc="upper right", framealpha=0.85)

    for q in range(C, nrow * ncol):
        sub.add_subplot(gs[q // ncol, q % ncol]).axis("off")

    if title:
        sub.suptitle(title, fontsize=10.5, fontweight="bold", y=0.99)


# ---------------------------------------------------------------------------
# Block G : cross-frequency coupling loss bar chart
# ---------------------------------------------------------------------------
# Off-diagonal band-pair indices, labels, and colors (source → target band).
_OD_BAND_PAIRS  = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
_OD_BAND_LABELS = ["L\u2192M", "L\u2192S", "M\u2192L",
                   "M\u2192S", "S\u2192L", "S\u2192M"]
_OD_BAND_COLORS = ["#5b5bd6", "#a0aee8", "#2e8b57",
                   "#7ec8a0", "#c0392b", "#e89090"]


def draw_crossfreq_loss_block(
    sub,
    Q_gt: np.ndarray,           # (n_pairs, 3, 3)
    Q_ffm: np.ndarray,          # (n_pairs, 3, 3)
    pairs: List[Tuple[int, int]],
    names: Sequence[str],
    title: Optional[str] = None,
) -> None:
    """Stacked bar chart of L_crossfreq per field pair, broken down by band pair.

    Each bar shows the six off-diagonal band-pair contributions (L→M, L→S,
    M→L, M→S, S→L, S→M), sorted by descending total cross-frequency loss.
    """
    diff2 = (Q_ffm - Q_gt) ** 2                         # (n_pairs, 3, 3)
    n_pairs = len(pairs)
    loss_bpb = np.zeros((len(_OD_BAND_PAIRS), n_pairs))
    for k, (bi, bj) in enumerate(_OD_BAND_PAIRS):
        loss_bpb[k] = diff2[:, bi, bj]                  # (n_pairs,)
    total_pp = loss_bpb.sum(axis=0)                      # (n_pairs,)
    order = np.argsort(-total_pp)
    plabels = [pair_label(names, c1, c2) for c1, c2 in pairs]
    x = np.arange(n_pairs)
    bottom = np.zeros(n_pairs)

    gs = sub.add_gridspec(1, 1, left=0.08, right=0.97, top=0.84, bottom=0.22)
    ax = sub.add_subplot(gs[0, 0])
    for k in range(len(_OD_BAND_PAIRS)):
        ax.bar(x, loss_bpb[k][order], bottom=bottom, width=0.74,
               color=_OD_BAND_COLORS[k], label=_OD_BAND_LABELS[k],
               edgecolor="white", lw=0.4)
        bottom += loss_bpb[k][order]
    ax.set_xticks(x)
    ax.set_xticklabels([plabels[k] for k in order], rotation=55, ha="right",
                       fontsize=6.6)
    ax.set_ylabel("$L_{\\mathrm{crossfreq}}^{c_1 c_2}$  (FFM vs GT)",
                  fontsize=8.5)
    ax.set_title("Cross-frequency loss by field pair and band transition",
                 fontsize=9)
    ax.legend(fontsize=7.0, title="band transition", title_fontsize=7.0,
              loc="upper right", ncol=2)
    ax.margins(x=0.01)

    if title:
        sub.suptitle(title, fontsize=10.5, fontweight="bold", y=0.99)


# ---------------------------------------------------------------------------
# Block H : coherence residual profile
# ---------------------------------------------------------------------------
def draw_coherence_residual_block(
    sub,
    basis: GraphSpectralBasis,
    est_gt: SpectralEstimate,
    est_ffm: SpectralEstimate,
    pairs: List[Tuple[int, int]],
    highlight_pair_idx: int,
    names: Sequence[str],
    cfg: VizConfig,
    n_show: int = 6,
    title: Optional[str] = None,
) -> None:
    """Signed coherence residual Δγ²(ν) = γ²_FFM(ν) - γ²_GT(ν) per pair.

    Shows the top n_show pairs by total squared residual power, always
    including the highlighted pair.  A zero reference line and band shading
    make systematic over/under-coherence easy to identify across frequencies.
    """
    bin_assign, centers = frequency_bins(basis, cfg.n_freq_bins)
    g_gt = coherence_from_bins(est_gt, bin_assign)       # (n_bins, n_pairs)
    g_ff = coherence_from_bins(est_ffm, bin_assign)
    delta = g_ff - g_gt                                  # (n_bins, n_pairs)

    resid_power = (delta ** 2).sum(axis=0)               # (n_pairs,)
    sorted_order = np.argsort(-resid_power)
    show = list(sorted_order[:n_show])
    if highlight_pair_idx not in show:
        show[-1] = highlight_pair_idx                    # always include primary pair

    gs = sub.add_gridspec(1, 1, left=0.09, right=0.97, top=0.84, bottom=0.18)
    ax = sub.add_subplot(gs[0, 0])
    _shade_bands(ax, basis.band_edges)
    ax.axhline(0, color="gray", lw=0.9, ls="--", zorder=2)

    colors = plt.cm.tab10(np.linspace(0, 0.9, len(show)))
    for k, p in enumerate(show):
        c1, c2 = pairs[p]
        lw = 2.0 if p == highlight_pair_idx else 1.2
        ax.plot(centers, delta[:, p], color=colors[k], lw=lw, zorder=3,
                label=pair_label(names, c1, c2))

    ax.set_xlabel("graph frequency  $\\nu_k=\\sqrt{\\lambda_k}$", fontsize=8.5)
    ax.set_ylabel("$\\Delta\\gamma^2$  (FFM \u2013 GT)", fontsize=8.5)
    ax.set_xlim(centers.min(), centers.max())
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.85,
              title="field pair", title_fontsize=7.0)
    band_centers = [0.5 * (basis.band_edges[b] + basis.band_edges[b + 1])
                    for b in range(3)]
    for b, nm in enumerate(("large", "medium", "small")):
        ax.text(band_centers[b], 0.03, nm, ha="center", fontsize=7.0,
                color=BAND_EDGE_C[b], style="italic",
                transform=ax.get_xaxis_transform())

    if title:
        sub.suptitle(title, fontsize=10.5, fontweight="bold", y=0.99)


################################################################################
# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------
def make_all_figures(
    out_dir: str | Path,
    xy: np.ndarray,
    basis: GraphSpectralBasis,
    fields_gt_repr: np.ndarray,      # (N, C) one representative GT snapshot
    fields_ffm_repr: np.ndarray,     # (N, C) matching FFM reconstruction
    est_gt: SpectralEstimate,        # ensemble GT spectra
    est_ffm: SpectralEstimate,       # ensemble FFM spectra
    names: Sequence[str],
    pair: Optional[Tuple[int, int]] = None,
    cfg: Optional[VizConfig] = None,
    prefix: str = "cross_spectral_coherence",
) -> Dict[str, Path]:
    """Render the combined figure + individual panels and save them.

    If ``pair`` is None the most strongly coupled GT pair is auto-selected.
    Returns a dict of saved file paths.
    """
    cfg = cfg or VizConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ps.set_paper_style()

    C = fields_gt_repr.shape[1]
    gb_gt = band_coherence(est_gt, basis)                # (3, n_pairs)
    gb_ff = band_coherence(est_ffm, basis)

    if pair is None:
        pair_idx = int(np.argmax(gb_gt.mean(axis=0)))
        pair = est_gt.pairs[pair_idx]
    else:
        pair = tuple(pair)
        pair_idx = est_gt.pairs.index(pair)

    mat_gt = coherence_matrix(gb_gt, est_gt.pairs, C)
    mat_ff = coherence_matrix(gb_ff, est_ffm.pairs, C)
    mat_err = mat_ff - mat_gt
    loss = coherence_loss_breakdown(est_ffm, est_gt, basis, cfg.n_freq_bins)

    saved: Dict[str, Path] = {}

    # ---- combined figure (3 stacked sub-figures) ----
    fig = plt.figure(figsize=(13.0, 15.0))
    subs = fig.subfigures(3, 1, height_ratios=[1.05, 1.15, 0.85])
    draw_spatial_block(
        subs[0], xy, basis, fields_gt_repr, fields_ffm_repr, pair, names,
        gb_gt[:, pair_idx], gb_ff[:, pair_idx], cfg,
        title=f"a   Scale-resolved coupling: {pair_label(names, *pair)}")
    draw_matrix_block(
        subs[1], mat_gt, mat_err, names,
        title="b   Cross-field coherence across all pairs, per scale")
    draw_curve_block(
        subs[2], basis, est_gt, est_ffm, pair_idx, pair, names, loss, cfg,
        title="c   Coherence spectra and loss decomposition")
    p = out_dir / f"{prefix}_combined.png"
    fig.savefig(p, dpi=cfg.dpi, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    saved["combined_png"] = p
    saved["combined_pdf"] = p.with_suffix(".pdf")

    # ---- individual panels ----
    fa = plt.figure(figsize=(13.0, 5.2))
    draw_spatial_block(fa.subfigures(1, 1), xy, basis, fields_gt_repr,
                       fields_ffm_repr, pair, names, gb_gt[:, pair_idx],
                       gb_ff[:, pair_idx], cfg,
                       title=f"Scale-resolved coupling: {pair_label(names, *pair)}")
    pa = out_dir / f"{prefix}_panelA_spatial.png"
    fa.savefig(pa, dpi=cfg.dpi, bbox_inches="tight"); plt.close(fa)
    saved["panelA"] = pa

    fb = plt.figure(figsize=(9.5, 6.2))
    draw_matrix_block(fb.subfigures(1, 1), mat_gt, mat_err, names,
                      title="Cross-field coherence across all pairs, per scale")
    pb = out_dir / f"{prefix}_panelB_matrices.png"
    fb.savefig(pb, dpi=cfg.dpi, bbox_inches="tight"); plt.close(fb)
    saved["panelB"] = pb

    fc = plt.figure(figsize=(11.5, 4.2))
    draw_curve_block(fc.subfigures(1, 1), basis, est_gt, est_ffm, pair_idx,
                     pair, names, loss, cfg,
                     title="Coherence spectra and loss decomposition")
    pc = out_dir / f"{prefix}_panelC_curves.png"
    fc.savefig(pc, dpi=cfg.dpi, bbox_inches="tight"); plt.close(fc)
    saved["panelC"] = pc

    fd = figure_all_pair_curves(basis, est_gt, est_ffm, names, cfg)
    pd = out_dir / f"{prefix}_allpairs_curves.png"
    fd.savefig(pd, dpi=cfg.dpi, bbox_inches="tight"); plt.close(fd)
    saved["allpairs"] = pd

    return saved

################################################################################
# ---------------------------------------------------------------------------
# Cross-frequency coupling figure suite (panels D–H)
# ---------------------------------------------------------------------------
def make_crossfreq_figures(
    out_dir: str | Path,
    basis: GraphSpectralBasis,
    est_gt: SpectralEstimate,
    est_ffm: SpectralEstimate,
    be_gt: BandEnergyEstimate,
    be_ffm: BandEnergyEstimate,
    Q_gt: np.ndarray,            # (n_pairs, 3, 3) GT cross-band coupling
    Q_ffm: np.ndarray,           # (n_pairs, 3, 3) FFM cross-band coupling
    names: Sequence[str],
    pair: Optional[Tuple[int, int]] = None,
    cfg: Optional[VizConfig] = None,
    prefix: str = "cross_spectral_coherence",
) -> Dict[str, Path]:
    """Render panels D–H for cross-frequency coupling and save them.

    Also produces a combined figure (D + G stacked) for easy comparison.
    If ``pair`` is None the most cross-band-coupled GT pair is auto-selected.
    Returns a dict of saved file paths (keys: panelD … panelH,
    crossfreq_combined_png, crossfreq_combined_pdf).
    """
    cfg = cfg or VizConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ps.set_paper_style()

    pairs = est_gt.pairs
    offdiag = ~np.eye(3, dtype=bool)
    if pair is None:
        pair_idx = int(np.argmax(
            [(Q_gt[p] * offdiag).mean() for p in range(len(pairs))]))
        pair = pairs[pair_idx]
    else:
        pair = tuple(pair)
        pair_idx = pairs.index(pair) if pair in pairs else 0

    n_show_pairs = min(6, len(pairs))
    C = len(names)
    saved: Dict[str, Path] = {}

    # ---- Panel D : cross-band Q̂ matrices ----
    fd = plt.figure(figsize=(max(6.0, 2.8 * n_show_pairs), 5.5))
    draw_crossband_matrix_block(
        fd.subfigures(1, 1), Q_gt, Q_ffm, pairs, names,
        n_show=n_show_pairs,
        title="d   Cross-band coupling matrices "
              "$\\hat{Q}_{c_1 c_2}^{B_i B_j}$  (GT, top; FFM\u2013GT, bottom)")
    pd_path = out_dir / f"{prefix}_panelD_crossband_matrices.png"
    fd.savefig(pd_path, dpi=cfg.dpi, bbox_inches="tight"); plt.close(fd)
    saved["panelD"] = pd_path

    # ---- Panel E : band-energy scatter ----
    fe = plt.figure(figsize=(7.5, 7.5))
    draw_band_energy_scatter(
        fe.subfigures(1, 1), be_gt, be_ffm, pair, names,
        Q_gt[pair_idx], Q_ffm[pair_idx],
        title=f"e   Band-energy scatter: {pair_label(names, *pair)}")
    pe_path = out_dir / f"{prefix}_panelE_band_energy_scatter.png"
    fe.savefig(pe_path, dpi=cfg.dpi, bbox_inches="tight"); plt.close(fe)
    saved["panelE"] = pe_path

    # ---- Panel F : per-field auto-spectra ----
    ncol_f = min(4, C)
    nrow_f = int(np.ceil(C / ncol_f))
    ff = plt.figure(figsize=(3.0 * ncol_f, 2.5 * nrow_f))
    draw_auto_spectrum_block(
        ff.subfigures(1, 1), est_gt, est_ffm, basis, names, cfg,
        title="f   Per-field auto-spectrum $\\hat{p}_c[k]$  (GT \u2014, FFM --)")
    pf_path = out_dir / f"{prefix}_panelF_auto_spectra.png"
    ff.savefig(pf_path, dpi=cfg.dpi, bbox_inches="tight"); plt.close(ff)
    saved["panelF"] = pf_path

    # ---- Panel G : cross-frequency loss bars ----
    fg = plt.figure(figsize=(max(7.0, 0.9 * len(pairs)), 4.2))
    draw_crossfreq_loss_block(
        fg.subfigures(1, 1), Q_gt, Q_ffm, pairs, names,
        title="g   Cross-frequency coupling loss "
              "$L_{\\mathrm{crossfreq}}$ by field pair")
    pg_path = out_dir / f"{prefix}_panelG_crossfreq_loss.png"
    fg.savefig(pg_path, dpi=cfg.dpi, bbox_inches="tight"); plt.close(fg)
    saved["panelG"] = pg_path

    # ---- Panel H : coherence residual profile ----
    fh = plt.figure(figsize=(9.0, 4.2))
    draw_coherence_residual_block(
        fh.subfigures(1, 1), basis, est_gt, est_ffm, pairs, pair_idx,
        names, cfg,
        title="h   Coherence error profile  "
              "$\\Delta\\gamma^2(\\nu) = \\gamma^2_{\\mathrm{FFM}} "
              "- \\gamma^2_{\\mathrm{GT}}$")
    ph_path = out_dir / f"{prefix}_panelH_coherence_residual.png"
    fh.savefig(ph_path, dpi=cfg.dpi, bbox_inches="tight"); plt.close(fh)
    saved["panelH"] = ph_path

    # ---- combined D + G figure ----
    fig = plt.figure(figsize=(max(10.0, 2.8 * n_show_pairs), 10.5))
    subs = fig.subfigures(2, 1, height_ratios=[1.05, 0.95])
    draw_crossband_matrix_block(
        subs[0], Q_gt, Q_ffm, pairs, names, n_show=n_show_pairs,
        title="d   Cross-band coupling matrices")
    draw_crossfreq_loss_block(
        subs[1], Q_gt, Q_ffm, pairs, names,
        title="g   Cross-frequency coupling loss by field pair")
    p_comb = out_dir / f"{prefix}_crossfreq_combined.png"
    fig.savefig(p_comb, dpi=cfg.dpi, bbox_inches="tight")
    fig.savefig(p_comb.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    saved["crossfreq_combined_png"] = p_comb
    saved["crossfreq_combined_pdf"] = p_comb.with_suffix(".pdf")

    # ---- New: per-field band energy ratio and fraction panels ----
    from coherence.visualization import (
        draw_band_energy_ratio_block as _draw_ber,
        draw_band_fraction_block as _draw_bef,
        make_crossband_ratio_panels as _make_cbr,
    )

    # Band energy ratio (pred / GT), per field per scale band.
    # be_gt.raw / be_ffm.raw are (B, 3, C); take ensemble mean -> (3, C).
    be_gt_mean  = be_gt.raw.mean(axis=0)
    be_ffm_mean = be_ffm.raw.mean(axis=0)

    fi = plt.figure(figsize=(12.0, 4.2))
    _draw_ber(
        fi.subfigures(1, 1),
        be_gt_mean, be_ffm_mean, names,
        title="i   Per-field band energy ratio  (pred / GT)",
    )
    pi_path = out_dir / f"{prefix}_panelI_band_energy_ratio.png"
    fi.savefig(pi_path, dpi=cfg.dpi, bbox_inches="tight")
    fi.savefig(pi_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fi)
    saved["panelI_band_energy_ratio"] = pi_path
    saved["panelI_band_energy_ratio_pdf"] = pi_path.with_suffix(".pdf")

    # Band energy fraction comparison (GT vs model), per field.
    fj = plt.figure(figsize=(10.0, 4.2))
    _draw_bef(
        fj.subfigures(1, 1),
        be_gt_mean, be_ffm_mean, names,
        title="j   Band energy fraction  (GT vs model)",
    )
    pj_path = out_dir / f"{prefix}_panelJ_band_fraction.png"
    fj.savefig(pj_path, dpi=cfg.dpi, bbox_inches="tight")
    fj.savefig(pj_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fj)
    saved["panelJ_band_fraction"] = pj_path
    saved["panelJ_band_fraction_pdf"] = pj_path.with_suffix(".pdf")

    # Cross-band energy ratio |S_pred|/|S_GT|, per field pair.
    S_gt_full  = compute_S_matrix_from_be(be_gt)    # (3, 3, C, C)
    S_ffm_full = compute_S_matrix_from_be(be_ffm)
    saved.update(
        _make_cbr(
            out_dir,
            S_gt_full,
            S_ffm_full,
            names,
            pairs,
            cfg=None,
            prefix=prefix,
            stacked=True,
        )
    )

    return saved


################################################################################
# ---------------------------------------------------------------------------
# Extra spatial panels for additional field pairs
# ---------------------------------------------------------------------------
def make_pair_spatial_panels(
    out_dir: str | Path,
    xy: np.ndarray,
    basis: GraphSpectralBasis,
    fields_gt_repr: np.ndarray,
    fields_ffm_repr: np.ndarray,
    est_pairs: List[Tuple[int, int]],
    gb_gt: np.ndarray,               # (3, n_pairs) band coherence, GT
    gb_ffm: np.ndarray,              # (3, n_pairs) band coherence, FFM
    names: Sequence[str],
    pairs: Sequence[Tuple[int, int]],
    cfg: Optional[VizConfig] = None,
    prefix: str = "cross_spectral_coherence",
    stacked: bool = True,
) -> Dict[str, Path]:
    """Render the spatial co-fluctuation block (panel A) for several field pairs.

    Reuses :func:`draw_spatial_block` so the panels are identical in style to the
    auto-selected one in :func:`make_all_figures`. Each pair is saved on its own,
    and (when ``stacked`` and >1 pair) also as a single vertically stacked figure
    for side-by-side comparison. ``pairs`` entries are (i, j) field indices and
    must exist in ``est_pairs``.
    """
    cfg = cfg or VizConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ps.set_paper_style()
    pairs = [tuple(p) for p in pairs]

    def _pidx(pair: Tuple[int, int]) -> int:
        try:
            return est_pairs.index(pair)
        except ValueError:
            raise ValueError(
                f"pair {pair} ({pair_label(names, *pair)}) not in est.pairs; "
                "field indices must be ascending and within range.")

    saved: Dict[str, Path] = {}

    # ---- one figure per pair ----
    for pair in pairs:
        idx = _pidx(pair)
        fig = plt.figure(figsize=(13.0, 5.2))
        draw_spatial_block(
            fig.subfigures(1, 1), xy, basis, fields_gt_repr, fields_ffm_repr,
            pair, names, gb_gt[:, idx], gb_ffm[:, idx], cfg,
            title=f"Scale-resolved coupling: {pair_label(names, *pair)}")
        p = out_dir / f"{prefix}_pair_{names[pair[0]]}_{names[pair[1]]}_spatial.png"
        fig.savefig(p, dpi=cfg.dpi, bbox_inches="tight"); plt.close(fig)
        saved[f"pair_{names[pair[0]]}_{names[pair[1]]}"] = p

    # ---- stacked multi-pair figure ----
    if stacked and len(pairs) > 1:
        fig = plt.figure(figsize=(13.0, 5.0 * len(pairs)))
        subs = fig.subfigures(len(pairs), 1)
        for sub, pair in zip(np.atleast_1d(subs), pairs):
            idx = _pidx(pair)
            draw_spatial_block(
                sub, xy, basis, fields_gt_repr, fields_ffm_repr, pair, names,
                gb_gt[:, idx], gb_ffm[:, idx], cfg,
                title=f"Scale-resolved coupling: {pair_label(names, *pair)}")
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
        "Cross-spectral coherence figure for turbulent combustion.")
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
                   help="Explicit split positions for the ensemble (overrides "
                        "--ensemble-size).")
    p.add_argument("--repr-index", type=int, default=None,
                   help="Split position used for the spatial maps (default: "
                        "first ensemble member).")
    p.add_argument("--cond-fields", type=int, nargs="*", default=None,
                   help="Conditioned field indices (default from run config / T).")
    p.add_argument("--n-obs", type=int, default=256,
                   help="Sensors per conditioned field.")
    p.add_argument("--n-steps", type=int, default=50,
                   help="Generative ODE steps.")
    p.add_argument("--ode-solver", type=str, default=None)
    # graph-spectral basis
    p.add_argument("--n-modes", type=int, default=384,
                   help="Graph-Laplacian eigenmodes (incl. DC).")
    p.add_argument("--k-neighbors", type=int, default=12)
    p.add_argument("--sigma-scale", type=float, default=1.0)
    p.add_argument("--band-mode", type=str, default="freq_equal",
                   choices=["freq_equal", "quantile"])
    p.add_argument("--n-freq-bins", type=int, default=24,
                   help="Frequency bins for the coherence-spectrum curves.")
    p.add_argument("--basis-cache", type=str, default=None,
                   help="Path to cache/reuse the eigenbasis (.npz).")
    # figure
    p.add_argument("--pair", type=int, nargs=2, default=None,
                   help="Field indices for the primary spatial block (default: "
                        "most coherent GT pair).")
    p.add_argument("--extra-pairs", type=str, nargs="*", default=None,
                   help="Additional field pairs to render as spatial panels, "
                        "given by name or index, e.g. --extra-pairs CO,T CH4,CO "
                        "(equivalently 1,2 0,1).")
    p.add_argument("--top-pairs", type=int, default=0,
                   help="Also render spatial panels for the N most coherent GT "
                        "pairs (deduplicated against --pair / --extra-pairs).")
    p.add_argument("--save-stack", type=str, default=None,
                   help="Directory to dump the reconstructed ensemble as "
                        "per-snapshot .npz files (coords/truth/recon/names) so "
                        "any pair can be re-rendered later via --npz-stack with "
                        "no GPU.")
    p.add_argument("--marker-size", type=float, default=2.2)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pair parsing
# ---------------------------------------------------------------------------
def resolve_pair(token: str, names: Sequence[str]) -> Tuple[int, int]:
    """Parse one 'a,b' token (field names or indices) into an ascending (i, j)."""
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


def save_stack_npz(stack_dir: Path, coords: np.ndarray,
                   gt_stack: Sequence[np.ndarray], ffm_stack: Sequence[np.ndarray],
                   names: Sequence[str]) -> None:
    """Dump the reconstructed ensemble in the per-snapshot format load_from_npz
    reads, so pairs can be re-rendered later via --npz-stack without a GPU."""
    stack_dir.mkdir(parents=True, exist_ok=True)
    fn = np.asarray(list(names))
    for s, (g, f) in enumerate(zip(gt_stack, ffm_stack)):
        np.savez_compressed(
            stack_dir / f"snap_{s:03d}.npz",
            coords_xy=np.asarray(coords), truth_phys=np.asarray(g),
            recon_phys=np.asarray(f), field_names=fn)
    print(f"[csc] saved reconstructed stack ({len(gt_stack)} snaps) -> "
          f"{stack_dir}  (re-render via --npz-stack '{stack_dir}/*.npz')")


# ---------------------------------------------------------------------------
# Offline NPZ mode
# ---------------------------------------------------------------------------
def load_from_npz(paths: Sequence[str]
                  ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], List[str]]:
    """Return coords (N,3), GT stack, FFM stack, field names from .npz files."""
    files = []
    for pat in paths:
        files.extend(sorted(Path(p) for p in glob.glob(pat))
                     if any(c in pat for c in "*?[") else [Path(pat)])
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
    print(f"[csc] loaded {len(gt)} snapshots from npz; N={coords.shape[0]}, "
          f"fields={names}")
    return coords, gt, ffm, names


# ---------------------------------------------------------------------------
# Model mode
# ---------------------------------------------------------------------------
def _reconcile_arch_with_checkpoint(cfg: dict, state_dict: dict) -> dict:
    """Infer architecture flags the saved config may omit, from checkpoint shapes.

    Legacy GL_rbf runs were sometimes trained with ``--sensor-coord-encoding
    fourier`` without persisting that key, so the config alone rebuilds the wrong
    sensor projection (raw coords -> in_dim 132 vs the checkpoint's fourier 513).
    We read ``field_embed`` / ``sensor_in_proj`` shapes and set the flags so
    build_model reconstructs the exact trained architecture.
    """
    cfg = dict(cfg)
    coord_dim = 3

    def _shape(name):
        for k in (name, f"model.{name}"):
            if k in state_dict:
                return tuple(state_dict[k].shape)
        return None

    fe = _shape("field_embed.weight")           # [n_fields, field_embed_dim]
    sp = _shape("sensor_in_proj.0.weight")      # [latent_dim, coord(+pe)+1+embed]
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
        print(f"[csc] checkpoint sensor_in_proj in_dim={sp[1]} -> "
              f"sensor_coord_encoding={cfg.get('sensor_coord_encoding')}, "
              f"use_fourier_pe={cfg.get('use_fourier_pe')}, "
              f"pe_num_bands={cfg.get('pe_num_bands')}")
    return cfg


def _load_baseline_and_dataset(args, cfg, run_dir, device):
    """Load a deterministic baseline (senseiver / mlp_rbf / geofno) checkpoint.

    Uses evaluate_coherence.load_baseline_context so the bundle is built and
    weights are loaded exactly as in the baseline evaluation scripts.  Returns
    the same 5-tuple as load_model_and_dataset so the rest of the pipeline is
    unchanged.
    """
    import types
    import evaluate_coherence as EC

    # Allow --data CLI override to propagate into the nested baseline config.
    if args.data:
        cfg.setdefault("shared", {}).setdefault("paths", {})
        cfg["shared"]["paths"]["data_path"] = args.data

    # Build a minimal shim namespace that load_baseline_context expects.
    shim = types.SimpleNamespace(
        baseline_model="auto",          # overwritten from config inside the call
        split=args.split,
        n_steps=getattr(args, "n_steps", None),
        ode_solver=getattr(args, "ode_solver", None),
    )

    _, _, cfg, dataset, bundle, _, _n_steps, _ode_solver = EC.load_baseline_context(
        args=shim, checkpoint_arg=args.checkpoint, device=device
    )

    cond_cfg = cfg.get("shared", {}).get("conditioning", {})
    if args.cond_fields is not None:
        cond_fields = list(args.cond_fields)
    else:
        cf = cond_cfg.get("vis_cond_fields") or cond_cfg.get("cond_fields") or [2]
        cond_fields = EC.ensure_list(cf)

    recon_fn = EC.get_reconstruction_fn(cfg["baseline_model"])
    return bundle, dataset, device, cond_fields, recon_fn


def load_model_and_dataset(args):
    """Mirror evaluate_coherence.main() setup for the pointcloud-FFM model."""
    import torch
    import yaml as _yaml
    import evaluate_coherence as EC

    EC.set_seed(args.seed)
    checkpoint_path, run_dir = EC.choose_checkpoint(args.checkpoint)
    demo_dir = EC.infer_demo_dir(run_dir)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available()
                                          else "cpu"))

    # Peek at run_config.yaml BEFORE calling load_run_config.
    # load_run_config is FFM-specific: it falls through and raises FileNotFoundError
    # (looking for args.json) whenever the YAML contains a baseline_model key.
    _run_cfg_path = run_dir / "run_config.yaml"
    _peek_cfg: dict = {}
    if _run_cfg_path.exists():
        with open(_run_cfg_path, "r", encoding="utf-8") as _f:
            _peek_cfg = _yaml.safe_load(_f) or {}
    _baseline_model = str(_peek_cfg.get("baseline_model", "")).strip().lower() or None

    if _baseline_model in {"latent_fm", "s3gm", "sit"}:
        raise NotImplementedError(
            f"Generative baseline {_baseline_model!r} is not supported in this script. "
            "Use evaluate_coherence.py for generative baselines.")
    if _baseline_model in {"senseiver", "mlp_rbf", "geofno"}:
        return _load_baseline_and_dataset(args, _peek_cfg, run_dir, device)

    cfg = EC.load_run_config(run_dir)

    data_path = args.data or cfg.get("data")
    if data_path is None:
        raise ValueError("Dataset path missing; pass --data.")
    data_path = EC.resolve_input_path(str(data_path), label="Dataset",
                                      extra_base_dirs=[run_dir, demo_dir])

    stats_path = run_dir / "dataset_stats.pt"
    dataset = EC.TurbulentCombustionH5Dataset(
        h5_path=str(data_path), split=args.split,
        train_ratio=float(cfg.get("train_ratio", 0.9)),
        seed=int(cfg.get("seed", 42)),
        time_stride=int(cfg.get("time_stride", 1)),
        stats_path=str(stats_path) if stats_path.exists() else None,
    )

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except pickle.UnpicklingError:
        checkpoint = torch.load(checkpoint_path, map_location=device,
                                weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) \
        and "model" in checkpoint else checkpoint
    if isinstance(state_dict, dict) and "_metadata" in state_dict:
        state_dict = {k: v for k, v in state_dict.items() if k != "_metadata"}

    cfg = _reconcile_arch_with_checkpoint(cfg, state_dict)
    model = EC.build_model(cfg, dataset).to(device)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        print(f"[csc] strict load failed; retrying strict=False.\n  {str(exc)[:300]}")
        info = model.load_state_dict(state_dict, strict=False)
        n_miss, n_unexp = len(info.missing_keys), len(info.unexpected_keys)
        if n_miss or n_unexp:
            print(f"[csc] non-strict load: {n_miss} missing, {n_unexp} unexpected "
                  "keys (architecture may still differ from the checkpoint).")
    model.eval()

    # Conditioning defaults.
    if args.cond_fields is not None:
        cond_fields = list(args.cond_fields)
    else:
        cf = cfg.get("vis_cond_fields") or cfg.get("cond_fields") \
            or [cfg.get("cond_field", 2)]
        cond_fields = EC.ensure_list(cf)
    recon_fn = EC.get_reconstruction_fn(_baseline_model or "pointcloud_ffm")
    return model, dataset, device, cond_fields, recon_fn


def reconstruct_stack(model, dataset, device, recon_fn, positions,
                      cond_fields, n_obs, n_steps, ode_solver):
    """Reconstruct each position; return GT stack, FFM stack (lists of (N,C))."""
    import torch
    sig = inspect.signature(recon_fn)
    gt, ffm = [], []
    for pos in positions:
        kwargs = dict(model=model, dataset=dataset, device=device,
                      snapshot_index=int(pos), cond_fields=cond_fields,
                      n_obs_list=[n_obs], n_steps=n_steps)
        if "ode_solver" in sig.parameters:
            kwargs["ode_solver"] = ode_solver
        with torch.no_grad():
            out = recon_fn(**kwargs)
        t = out["truth"].detach().float().cpu().numpy()[0]   # (N, C)
        r = out["recon"].detach().float().cpu().numpy()[0]
        gt.append(t)
        ffm.append(r)
        print(f"[csc] reconstructed split-position {pos} "
              f"({len(gt)}/{len(positions)})")
    coords = dataset.coords_raw.detach().cpu().numpy().astype(np.float64)
    return coords, gt, ffm, list(dataset.field_names)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def write_summary(path: Path, names, pairs, gb_gt, gb_ffm, loss, selected_pair,
                  basis, meta):
    band_names = ["large", "medium", "small"]
    plabels = [pair_label(names, i, j) for (i, j) in pairs]
    summary = {
        "selected_pair": pair_label(names, *selected_pair),
        "field_names": list(names),
        "n_modes": int(basis.n_modes),
        "band_edges_freq": [float(x) for x in basis.band_edges],
        "modes_per_band": [int((basis.band_of_mode == b).sum()) for b in range(3)],
        "coherence_loss_total": float(loss["total"]),
        "coherence_loss_band_fraction": {
            band_names[b]: float(loss["band_fraction"][b]) for b in range(3)},
        "per_pair": {},
        "meta": meta,
    }
    for p, lab in enumerate(plabels):
        summary["per_pair"][lab] = {
            "gamma2_gt": {band_names[b]: float(gb_gt[b, p]) for b in range(3)},
            "gamma2_ffm": {band_names[b]: float(gb_ffm[b, p]) for b in range(3)},
            "coh_loss": float(loss["per_pair"][p]),
            "coh_loss_by_band": {band_names[b]: float(loss["per_pair_band"][b, p])
                                 for b in range(3)},
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
        out_root = Path(args.out_dir or (SRC_DIR.parent / "Save_PhyCoEval"
                        / "cross_spectral_coherence" / ts))
        meta = {"mode": "npz", "n_snapshots": len(gt_stack)}
    else:
        if not args.checkpoint:
            raise SystemExit("Provide --checkpoint (model mode) or --npz-stack.")
        model, dataset, device, cond_fields, recon_fn = \
            load_model_and_dataset(args)
        if args.snapshot_indices:
            positions = list(args.snapshot_indices)
        else:
            m = min(args.ensemble_size, len(dataset))
            positions = sorted(set(np.linspace(0, len(dataset) - 1, m)
                                   .astype(int).tolist()))
        coords, gt_stack, ffm_stack, names = reconstruct_stack(
            model, dataset, device, recon_fn, positions, cond_fields,
            args.n_obs, args.n_steps, args.ode_solver)
        demo_dir = SRC_DIR.parent
        out_root = Path(args.out_dir or (demo_dir / "Save_PhyCoEval"
                        / "cross_spectral_coherence" / ts))
        meta = {"mode": "model", "checkpoint": str(args.checkpoint),
                "split": args.split, "cond_fields": list(cond_fields),
                "n_obs": args.n_obs, "n_steps": args.n_steps,
                "positions": [int(x) for x in positions]}

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.save_stack:
        save_stack_npz(Path(args.save_stack), coords, gt_stack, ffm_stack, names)

    # Representative snapshot for the spatial maps.
    repr_pos = 0 if args.repr_index is None else int(args.repr_index)
    repr_pos = min(repr_pos, len(gt_stack) - 1)
    fields_gt_repr = gt_stack[repr_pos]
    fields_ffm_repr = ffm_stack[repr_pos]

    # Graph-spectral basis.
    cache = args.basis_cache or str(out_root.parent / "graph_basis_cache.npz")
    basis = compute_graph_basis(
        coords, n_modes=args.n_modes, k_neighbors=args.k_neighbors,
        sigma_scale=args.sigma_scale, band_mode=args.band_mode,
        cache_path=cache, verbose=True)
    xy = coords[:, basis.keep_axes]

    # Ensemble coherence.
    est_gt = accumulate_spectra(basis, gt_stack)
    est_ffm = accumulate_spectra(basis, ffm_stack)
    gb_gt = band_coherence(est_gt, basis)
    gb_ffm = band_coherence(est_ffm, basis)
    loss = coherence_loss_breakdown(est_ffm, est_gt, basis, args.n_freq_bins)

    ranking = rank_pairs_by_coherence(est_gt, basis)
    print("[csc] GT pair ranking (most coherent first): "
          + ", ".join(pair_label(names, *est_gt.pairs[k]) for k in ranking[:5]))

    cfg = VizConfig(marker_size=args.marker_size, n_freq_bins=args.n_freq_bins)
    pair = tuple(args.pair) if args.pair else None
    selected = pair or est_gt.pairs[int(np.argmax(gb_gt.mean(axis=0)))]
    saved = make_all_figures(
        out_root, xy, basis, fields_gt_repr, fields_ffm_repr, est_gt, est_ffm,
        names, pair=pair, cfg=cfg)

################################################################################

    # Cross-frequency coupling figures (panels D–H).
    print("[csc] computing cross-frequency coupling ...")
    be_gt  = compute_band_energies(basis, gt_stack)
    be_ffm = compute_band_energies(basis, ffm_stack)
    Q_gt   = cross_band_coupling_scores(be_gt,  est_gt.pairs)
    Q_ffm  = cross_band_coupling_scores(be_ffm, est_ffm.pairs)
    saved.update(make_crossfreq_figures(
        out_root, basis, est_gt, est_ffm, be_gt, be_ffm, Q_gt, Q_ffm,
        names, pair=selected, cfg=cfg))
################################################################################
    # Extra spatial panels for additional pairs (explicit and/or top-N).
    extra: List[Tuple[int, int]] = []
    if args.extra_pairs:
        extra += [resolve_pair(tok, names) for tok in args.extra_pairs]
    if args.top_pairs > 0:
        extra += [tuple(sorted(est_gt.pairs[int(k)]))
                  for k in ranking[:args.top_pairs]]
    seen, extra_unique = set(), []
    for pr in extra:
        if pr not in seen:
            seen.add(pr)
            extra_unique.append(pr)
    if extra_unique:
        print("[csc] extra spatial panels: "
              + ", ".join(pair_label(names, *pr) for pr in extra_unique))
        saved.update(make_pair_spatial_panels(
            out_root, xy, basis, fields_gt_repr, fields_ffm_repr, est_gt.pairs,
            gb_gt, gb_ffm, names, extra_unique, cfg=cfg))

    write_summary(out_root / "cross_spectral_coherence_summary.json", names,
                  est_gt.pairs, gb_gt, gb_ffm, loss, selected, basis, meta)

    print("\n[csc] done. Figures saved:")
    for k, v in saved.items():
        print(f"    {k:14s} {v}")


if __name__ == "__main__":
    main()
