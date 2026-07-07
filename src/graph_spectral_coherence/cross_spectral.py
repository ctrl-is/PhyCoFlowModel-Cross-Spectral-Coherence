from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass

def gft(fields: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """
    Project fields into graph-frequency space.

    Args:
        fields: [B, N, C] or [B, Q, C].
        U: [N, K] full basis or [B, Q, K] sampled basis.

    Returns:
        [B, K, C] graph Fourier coefficients.
    """
    if U.ndim == 2:
        N_u = U.shape[0]
        N_f = fields.shape[1]
        if N_u != N_f:
            raise ValueError(
                f"Spatial dimension mismatch: U has {N_u} nodes but fields have {N_f} points. The GFT requires the full unsubsampled field."
            )
        return torch.einsum("nk,bnc->bkc", U, fields)
    
    if U.ndim == 3:
        B_u, N_u, _ = U.shape
        B_f, N_f, _ = fields.shape
        if B_u != B_f or N_u != N_f:
            raise ValueError(
                f"Batched U mismatch: U has shape {tuple(U.shape)}, fields have shape {tuple(fields.shape)}."
            )
        return torch.einsum("bnk,bnc->bkc", U, fields)
    
    raise ValueError(f"U must have shape [N, K] or [B, N, K], got {tuple(U.shape)}.")

def inverse_graph_fourier_transform(gft_coeffs: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
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

def estimate_auto_spectra(gft_coeffs: torch.Tensor) -> torch.Tensor:
    """
    Graph power spectral density: the energy distribution per field per frequency.

    hat{p}_c[k] = (1/B) sum_b |hat{X}_{b,k,c}|^2

    Args:
        gft_coeffs: [B, K, C] - torch.Tensor

    Returns:
        [K, C] — auto_spectra[k, c] = average energy of field c at frequency k
    """
    return (torch.abs(gft_coeffs) ** 2).mean(dim=0)

def estimate_cross_spectra(gft_coeffs: torch.Tensor) -> torch.Tensor:
    """
    Graph cross-spectral density: the shared structure between field pairs.

    hat{p}_{c1,c2}[k] = (1/B) sum_b hat{X}_{b,k,c1} * conj(hat{X}_{b,k,c2})

    Args:
        gft_coeffs: [B, K, C] - torch.Tensor

    Returns:
        [K, C, C] — cross_spectra[k, i, j] = shared structure between
                    fields i and j at frequency k.
                    Diagonal entries equal the auto-spectra.
    """
    B = gft_coeffs.shape[0]
    return torch.einsum("bki,bkj->kij", gft_coeffs, torch.conj(gft_coeffs)) / B

def compute_coherence(auto_spectra: torch.Tensor, cross_spectra: torch.Tensor, eps: float = 1e-8,) -> torch.Tensor:
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
    numer = torch.abs(cross_spectra) ** 2
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
    eps_ratio: float = 1e-12

    # Loss weights
    # lambda_coh/lambda_cross/lambda_auto are legacy auxiliary-loss weights
    # eta_crossfreq is for the updated physical coherence objective
    lambda_coh: float = 1.0
    lambda_cross: float = 1.0
    lambda_auto: float = 1.0
    eta_crossfreq: float = 1.0

    # Optional: restrict to specific field pairs (None = all pairs i < j), List of tuples
    field_pairs: any = None

def same_freq_coherence_loss(
        coherence_pred: torch.Tensor, 
        coherence_target: torch.Tensor, 
        field_pairs: list[Tuple[int, int]] | None = None
        ) -> torch.Tensor:
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
        loss = loss + (torch.abs(diff) ** 2).mean()

    return loss / len(field_pairs)


def same_freq_cross_spectrum_loss(
        cross_spectra_pred: torch.Tensor, 
        cross_spectra_target: torch.Tensor , 
        field_pairs: list[Tuple[int, int]] | None = None
        ) -> torch.Tensor:
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
        loss = loss + (torch.abs(diff) ** 2).mean()

    return loss / len(field_pairs)


def same_freq_auto_spectrum_loss(auto_spectra_pred: torch.Tensor, auto_spectra_target: torch.Tensor) -> torch.Tensor:
    """
    L_auto = (1/C) sum_c || p^pred_c - p^data_c ||^2

    Ensures each field individually has the correct frequency content.
    """
    C = auto_spectra_pred.shape[1]
    return ((auto_spectra_pred - auto_spectra_target) ** 2).mean()


def legacy_combined_spectral_loss(
        fields_pred: torch.Tensor,
        fields_target: torch.Tensor,
        U: torch.Tensor,
        cfg: CrossSpectralConfig | None = None
        ) -> Dict[str, torch.Tensor]:
    """
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
    L_coh = same_freq_coherence_loss(coh_pred, coh_target, cfg.field_pairs)
    L_cross = same_freq_cross_spectrum_loss(cross_pred, cross_target, cfg.field_pairs)
    L_auto = same_freq_auto_spectrum_loss(auto_pred, auto_target)

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

def compute_same_freq_coherence_diagnostics(
        fields_pred: torch.Tensor,
        fields_target: torch.Tensor,
        U: torch.Tensor,
        bands: dict[str, list[int]] | np.ndarray | torch.Tensor | None = None,
        cfg: CrossSpectralConfig | None = None
        ) -> dict[str, object]:
    """
    Compute graph coherence loss and optional band decomposition.

    Args:
        fields_pred: [B, N, C] predicted full fields.
        fields_target: [B, N, C] target full fields.
        U: [N, K] graph Fourier basis.
        bands: Optional dict of frequency-band indices.
        cfg: Optional CrossSpectralConfig.

    Returns:
        Dict containing L_coh, diagnostics, and band losses.
    """
    cfg = cfg or CrossSpectralConfig()

    gft_pred = gft(fields_pred, U)
    gft_target = gft(fields_target, U)

    auto_pred = estimate_auto_spectra(gft_pred)
    auto_target = estimate_auto_spectra(gft_target)

    cross_pred = estimate_cross_spectra(gft_pred)
    cross_target = estimate_cross_spectra(gft_target)

    coh_pred = compute_coherence(auto_pred, cross_pred, eps=cfg.eps)
    coh_target = compute_coherence(auto_target, cross_target, eps=cfg.eps)

    L_coh = same_freq_coherence_loss(coh_pred, coh_target, cfg.field_pairs)

    L_cross = same_freq_cross_spectrum_loss(cross_pred, cross_target, cfg.field_pairs)
    L_auto = same_freq_auto_spectrum_loss(auto_pred, auto_target)

    # Optional low/mid/high decomposition of the same coherence loss.
    band_losses = {}
    band_ratios = {}

    if bands is not None:
        for band_name, band_idx in bands.items():
            band_idx = torch.as_tensor(
                band_idx,
                device=coh_pred.device,
                dtype=torch.long,
            )

            coh_pred_band = coh_pred[band_idx, :, :]
            coh_target_band = coh_target[band_idx, :, :]

            L_band = same_freq_coherence_loss(
                    coh_pred_band,
                    coh_target_band,
                    cfg.field_pairs,
                    )

            band_losses[band_name] = L_band
            band_ratios[band_name] = L_band / (L_coh + cfg.eps_ratio)

    return {
        "loss": L_coh,
        "L_coh": L_coh,
        "L_cross": L_cross,
        "L_auto": L_auto,
        "band_losses": band_losses,
        "band_ratios": band_ratios,
        "coh_pred": coh_pred,
        "coh_target": coh_target,
    }

def compute_band_energy(
        gft_coeffs: torch.Tensor, 
        band_idx: list[int] | np.ndarray | torch.Tensor
        ) -> torch.Tensor:
    """
    Computes the band energy for one graph-frequency band.

    Args:
        gft_coeffs: [B, K, C]
        band_idx: indices of graph frequencies in the band

    Returns:
        band_energy: [B, C]
    """
    band_idx = torch.as_tensor(band_idx, device=gft_coeffs.device, dtype=torch.long)
    selected_coeffs = gft_coeffs[:, band_idx, :]
    # take the magnitude in case of complex values
    energy = torch.sum(torch.abs(selected_coeffs) ** 2, dim=1)
    
    return energy

def compute_centered_band_energy(band_energy: torch.Tensor) -> torch.Tensor:
    """
    Centers band energy across the batch dimension.

    Args:
        band_energy: [B, C]

    Returns:
        centered_band_energy: [B, C]
    """
    mean_energy = band_energy.mean(dim=0)
    centered_energy = band_energy - mean_energy
    
    return centered_energy

def compute_all_centered_band_energies(
        gft_coeffs: torch.Tensor,
        bands: Dict[str, list[int] | np.ndarray | torch.Tensor]
        ) -> Tuple[Dict[str, torch.Tensor], list[str]]:
    """
    Computes centered band energies for all frequency bands.

    Args:
        gft_coeffs: [B, K, C]
        bands: dict mapping band name -> frequency indices

    Returns:
        centered_band_energies: mapping band name -> centered band energy [B, C]
        band_names = list of band names in consistent order
    """
    centered_band_energies = {}
    band_names = list(bands.keys())

    for band_name in band_names:
        band_idx = bands[band_name]

        band_energy = compute_band_energy(gft_coeffs, band_idx)
        centered_energy = compute_centered_band_energy(band_energy)

        centered_band_energies[band_name] = centered_energy

    return centered_band_energies, band_names

def compute_all_cross_band_covariances(
        gft_coeffs: torch.Tensor,
        bands: dict[str, list[int] | np.ndarray | torch.Tensor],
        ) -> tuple[torch.Tensor, list[str]]:
    """
    Computes all cross-band covariance matrices.

    Args:
        gft_coeffs: [B, K, C]
        bands: dict mapping band name -> frequency indices

    Returns:
        S: [M, M, C, C]
        band_names: list[str]
    """
    centered_band_energies, band_names = compute_all_centered_band_energies(
        gft_coeffs,
        bands,
    )

    M = len(band_names)
    _, C = next(iter(centered_band_energies.values())).shape

    S = torch.empty(
        (M, M, C, C),
        device=gft_coeffs.device,
        dtype=gft_coeffs.dtype,
    )

    for i, band_i in enumerate(band_names):
        for j, band_j in enumerate(band_names):
            centered_i = centered_band_energies[band_i]  # [B, C]
            centered_j = centered_band_energies[band_j]  # [B, C]

            B = centered_i.shape[0]
            S[i, j] = torch.einsum("bc,bd->cd", centered_i, centered_j) / B

    return S, band_names

def compute_normalized_cross_band_coupling(S: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes the normalized cross-band coupling scores.

    Args:
        S: [M, M, C, C]

    Returns:
        Q: [M, M, C, C]
    """
    M, _, C, _ = S.shape
    self_cov = torch.empty((M, C), device=S.device, dtype=S.dtype)

    for i in range(M):
        self_cov[i] = torch.diagonal(S[i, i], dim1=0, dim2=1)

    denom = torch.einsum("ic,jd->ijcd", self_cov, self_cov) + eps

    Q = (torch.abs(S) ** 2) / denom
    return Q

def cross_frequency_coupling_loss(
        Q_pred: torch.Tensor,
        Q_target: torch.Tensor,
        field_pairs: list[Tuple[int, int]] | None = None
        ) -> torch.Tensor:
    """
    Computes L_crossfreq by comparing predicted and target normalized
    cross-band coupling scores on off-diagonal band pairs.

    Args:
        Q_pred: [M, M, C, C]
        Q_target: [M, M, C, C]
        field_pairs: optional list of field index pairs

    Returns:
        scalar loss
    """
    M, _, C, _ = Q_pred.shape

    if field_pairs is None:
        field_pairs = [(i, j) for i in range(C) for j in range(i + 1, C)]

    if len(field_pairs) == 0:
        return torch.tensor(0.0, device=Q_pred.device, dtype=Q_pred.dtype)

    off_diag_mask = ~torch.eye(M, dtype=torch.bool, device=Q_pred.device)

    loss = torch.tensor(0.0, device=Q_pred.device, dtype=Q_pred.dtype)

    for c1, c2 in field_pairs:
        diff = Q_pred[:, :, c1, c2] - Q_target[:, :, c1, c2]
        off_diag_diff = diff[off_diag_mask]
        loss = loss + (torch.abs(off_diag_diff) ** 2).mean()
    
    return loss / len(field_pairs)

def compute_physical_coherence_loss(
        fields_pred: torch.Tensor,
        fields_target: torch.Tensor,
        U: torch.Tensor,
        bands: Dict[str, list[int] | np.ndarray | torch.Tensor],
        cfg: CrossSpectralConfig | None = None,
        ) -> Dict[str, object]:
    """
    Computes the final physical coherence loss:

        L_phys_coh = L_same + eta_crossfreq * L_crossfreq

    where:
        L_same compares same-frequency graph coherence.
        L_crossfreq compares off-diagonal cross-band coupling scores.

    Args:
        fields_pred: [B, N, C] predicted full fields
        fields_target: [B, N, C] target full fields
        U: [N, K] graph Fourier basis
        bands: dict mapping band name -> frequency-mode indices
        cfg: optional CrossSpectralConfig

    Returns:
        Dict containing total loss, component losses, and diagnostics.
    """
    cfg = cfg or CrossSpectralConfig()

    # Graph Fourier transforms: [B, N, C] -> [B, K, C]
    gft_pred = gft(fields_pred, U)
    gft_target = gft(fields_target, U)

    # ----- Same-frequency coherence loss: L_same -----
    auto_pred = estimate_auto_spectra(gft_pred)
    auto_target = estimate_auto_spectra(gft_target)

    cross_pred = estimate_cross_spectra(gft_pred)
    cross_target = estimate_cross_spectra(gft_target)

    coh_pred = compute_coherence(auto_pred, cross_pred, eps=cfg.eps)
    coh_target = compute_coherence(auto_target, cross_target, eps=cfg.eps)

    L_same = same_freq_coherence_loss(
        coh_pred,
        coh_target,
        cfg.field_pairs,
    )

    # ----- Cross-frequency coupling loss: L_crossfreq -----
    S_pred, band_names = compute_all_cross_band_covariances(gft_pred, bands)
    S_target, _ = compute_all_cross_band_covariances(gft_target, bands)

    Q_pred = compute_normalized_cross_band_coupling(S_pred, eps=cfg.eps)
    Q_target = compute_normalized_cross_band_coupling(S_target, eps=cfg.eps)

    L_crossfreq = cross_frequency_coupling_loss(
        Q_pred,
        Q_target,
        cfg.field_pairs,
    )

    # ----- Final physical coherence objective -----
    L_phys_coh = L_same + cfg.eta_crossfreq * L_crossfreq

    return {
        "loss": L_phys_coh,
        "L_phys_coh": L_phys_coh,
        "L_same": L_same,
        "L_crossfreq": L_crossfreq,

        # Same-frequency diagnostics
        "coh_pred": coh_pred.detach(),
        "coh_target": coh_target.detach(),
        "auto_spectra_pred": auto_pred.detach(),
        "auto_spectra_target": auto_target.detach(),
        "cross_spectra_pred": cross_pred.detach(),
        "cross_spectra_target": cross_target.detach(),

        # Cross-frequency diagnostics
        "S_pred": S_pred.detach(),
        "S_target": S_target.detach(),
        "Q_pred": Q_pred.detach(),
        "Q_target": Q_target.detach(),
        "band_names": band_names,
    }

