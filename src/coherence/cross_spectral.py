import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass


def gft(fields, U):
    """
    Project physical fields into graph-frequency space.

    Implements: hat{x}^(c) = U^T x^(c) for all fields and batch elements.

    The einsum contracts over the spatial dimension n:
        U is [N, K], fields is [B, N, C]
        Result is [B, K, C]

    Each output element gft[b, k, c] tells you the amplitude of the kth
    graph frequency mode in the c-th physical field of the b-th sample.

    Args:
        fields: [B, N, C] — batch of multi-field snapshots - torch.Tensor
        U:      [N, K]    — graph Fourier basis (computed previously, it is fixed) - torch.Tensor

    Returns:
        [B, K, C] — our graph Fourier coefficients
    """
    N_u = U.shape[0]
    N_f = fields.shape[1]
    if N_u != N_f:
        raise ValueError(
            f"Spatial dimension mismatch: U has {N_u} nodes but fields have {N_f} points. The GFT requires the full unsubsampled field."
            )
    return torch.einsum("nk,bnc->bkc", U, fields)

def inverse_graph_fourier_transform(gft_coeffs, U):
    """
    Reconstructs spatial fields from graph Fourier coefficients.

    Implements: x^(c) = U hat{x}^(c)

    This is the inverse of graph_fourier_transform. Useful for:
        - Verifying the transform is invertible (when K = N)
        - Visualizing what specific frequency bands look like in space
        - Band-pass filtering fields to isolate scale ranges

    Args:
        gft_coeffs: [B, K, C] — graph Fourier coefficients - torch.Tensor
        U:          [N, K]    — graph Fourier basis - torch.Tensor

    Returns:
        [B, N, C] — reconstructed spatial fields
    """
    return torch.einsum("nk,bkc->bnc", U, gft_coeffs)

def estimate_auto_spectra(gft_coeffs):
    """
    Graph power spectral density: the energy distribution per field per frequency.

    hat{p}_c[k] = (1/B) sum_b |hat{X}_{b,k,c}|^2

    Args:
        gft_coeffs: [B, K, C] - torch.Tensor

    Returns:
        [K, C] — auto_spectra[k, c] = average energy of field c at frequency k
    """
    return (gft_coeffs ** 2).mean(dim=0)
def estimate_cross_spectra(gft_coeffs):
    """
    Graph cross-spectral density: the shared structure between field pairs.

    hat{p}_{c1,c2}[k] = (1/B) sum_b hat{X}_{b,k,c1} * hat{X}_{b,k,c2}

    Args:
        gft_coeffs: [B, K, C] - torch.Tensor

    Returns:
        [K, C, C] — cross_spectra[k, i, j] = shared structure between
                    fields i and j at frequency k.
                    Diagonal entries equal the auto-spectra.
    """
    B = gft_coeffs.shape[0]
    return torch.einsum("bki,bkj->kij", gft_coeffs, gft_coeffs) / B

def compute_coherence(auto_spectra, cross_spectra, eps: float = 1e-8,):
    """
    Graph coherence: normalized spectral coupling strength.

    c_{XY}[k] = |p_{XY}[k]|^2 / (p_X[k] * p_Y[k] + eps)

    Result is in [0, 1] for each frequency and field pair.
        0 = fields are spectrally independent at this frequency
        1 = fields are perfectly coupled at this frequency

    Args:
        auto_spectra:  [K, C] - torch.Tensor
        cross_spectra: [K, C, C] - torch.Tensor
        eps: prevents division by zero

    Returns:
        [K, C, C] — coherence values in [0, 1]
    """
    denom = torch.einsum("ki,kj->kij", auto_spectra, auto_spectra)
    numer = cross_spectra ** 2
    return numer / (denom + eps)

@dataclass
class CrossSpectralConfig:
    """All hyperparameters for the cross-spectral coherence system."""

   # Graph construction (passed to graph.py functions)
    k_neighbors: int = 16
    sigma: float = None    # None = median heuristic
    num_modes: int = 256

    # Coherence computation
    eps: float = 1e-8

    # Loss weights (Section 4.7: L_spectral = λ_coh * L_coh + λ_cross * L_cross + λ_auto * L_auto)
    lambda_coh: float = 1.0
    lambda_cross: float = 1.0
    lambda_auto: float = 1.0

    # Optional: restrict to specific field pairs (None = all pairs i < j), List of tuples
    field_pairs: any = None


# Loss Functions (Sections 4.6-4.7)

def coherence_matching_loss(coherence_pred, coherence_target, field_pairs = None):
    """
    L_coh = (1/|pairs|) sum_{c1<c2} || c^pred_{c1,c2} - c^data_{c1,c2} ||^2

    coherence_pred & coherence_target both torch.Tensors

    Preserves the strength of cross-field coupling but not sign/phase.
    """
    C = coherence_pred.shape[1]
    if field_pairs is None:
        field_pairs = [(i, j) for i in range(C) for j in range(i + 1, C)]

    if len(field_pairs) == 0:
        return torch.tensor(0.0, device=coherence_pred.device, dtype=coherence_pred.dtype)

    loss = torch.tensor(0.0, device=coherence_pred.device, dtype=coherence_pred.dtype)
    for i, j in field_pairs:
        diff = coherence_pred[:, i, j] - coherence_target[:, i, j]
        loss = loss + (diff ** 2).mean()

    return loss / len(field_pairs)


def cross_spectrum_matching_loss(cross_spectra_pred, cross_spectra_target, field_pairs = None):
    """
    L_cross = (1/|pairs|) sum_{c1<c2} || p^pred_{c1,c2} - p^data_{c1,c2} ||^2

    Both cross_spectra_pred & cross_spectra_target both torch.Tensors

    Preserves sign/phase alignment between fields.
    """
    C = cross_spectra_pred.shape[1]
    if field_pairs is None:
        field_pairs = [(i, j) for i in range(C) for j in range(i + 1, C)]

    if len(field_pairs) == 0:
        return torch.tensor(0.0, device=cross_spectra_pred.device, dtype=cross_spectra_pred.dtype)

    loss = torch.tensor(0.0, device=cross_spectra_pred.device, dtype=cross_spectra_pred.dtype)
    for i, j in field_pairs:
        diff = cross_spectra_pred[:, i, j] - cross_spectra_target[:, i, j]
        loss = loss + (diff ** 2).mean()

    return loss / len(field_pairs)


def auto_spectrum_matching_loss(auto_spectra_pred,auto_spectra_target):
    """
    L_auto = (1/C) sum_c || p^pred_c - p^data_c ||^2

    Ensures each field individually has the correct frequency content.
    """
    C = auto_spectra_pred.shape[1]
    return ((auto_spectra_pred - auto_spectra_target) ** 2).mean()


def combined_spectral_loss(
    fields_pred: torch.Tensor,
    fields_target: torch.Tensor,
    U: torch.Tensor,
    cfg = None):
    """
    Full spectral regularizer (Section 4.7):
        L_spectral = λ_coh * L_coh + λ_cross * L_cross + λ_auto * L_auto

    REQUIREMENT: fields_pred and fields_target must be [B, N, C] where
    N matches U.shape[0]. These must be FULL spatial fields, not
    subsampled query points.

    Args:
        fields_pred:   [B, N, C] — model's reconstructed fields
        fields_target: [B, N, C] — ground truth fields
        U:             [N, K]    — precomputed graph Fourier basis
        cfg:           hyperparameters - optional

    Returns:
        Dict with 'total' loss and all intermediate quantities for logging.
    """
    cfg = cfg or CrossSpectralConfig()

    # Graph Fourier Transform
    gft_pred = gft(fields_pred, U)
    gft_target = gft(fields_target, U)

    # Estimate spectra from the batch
    auto_pred = estimate_auto_spectra(gft_pred)
    auto_target = estimate_auto_spectra(gft_target)

    cross_pred = estimate_cross_spectra(gft_pred)
    cross_target = estimate_cross_spectra(gft_target)

    # Coherence
    coh_pred = compute_coherence(auto_pred, cross_pred, eps=cfg.eps)
    coh_target = compute_coherence(auto_target, cross_target, eps=cfg.eps)

    # Losses
    L_coh = coherence_matching_loss(coh_pred, coh_target, cfg.field_pairs)
    L_cross = cross_spectrum_matching_loss(cross_pred, cross_target, cfg.field_pairs)
    L_auto = auto_spectrum_matching_loss(auto_pred, auto_target)

    L_total = (
        cfg.lambda_coh * L_coh
        + cfg.lambda_cross * L_cross
        + cfg.lambda_auto * L_auto
    )

    return {
        "total": L_total,
        "coh_loss": L_coh,
        "cross_loss": L_cross,
        "auto_loss": L_auto,
        "auto_spectra_pred": auto_pred.detach(),
        "auto_spectra_target": auto_target.detach(),
        "cross_spectra_pred": cross_pred.detach(),
        "cross_spectra_target": cross_target.detach(),
        "coherence_pred": coh_pred.detach(),
        "coherence_target": coh_target.detach(),
    }





