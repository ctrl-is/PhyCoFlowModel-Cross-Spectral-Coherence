# Code for defining cross-spectral coherence

# This will define parts 1.4-___ of our term.


import torch
import torch.nn as nn
import numpy as np


def gft(fields: torch.Tensor, U: torch.Tensor):
    """
    Project physical fields into graph-frequency space.

    Implements: hat{x}^(c) = U^T x^(c) for all fields and batch elements.

    The einsum contracts over the spatial dimension n:
        U is [N, K], fields is [B, N, C]
        Result is [B, K, C]

    Each output element gft[b, k, c] tells you the amplitude of the k-th
    graph frequency mode in the c-th physical field of the b-th sample.

    Args:
        fields: [B, N, C] — batch of multi-field snapshots
        U:      [N, K]    — graph Fourier basis (computed previously, fixed)

    Returns:
        [B, K, C] — our graph Fourier coefficients
    """
    N_u = U.shape[0]
    N_f = fields.shape[1]
    if N_u != N_f:
        raise ValueError(
            f"Spatial dimension mismatch: U has {N_u} nodes but fields "
            f"have {N_f} points. The GFT requires the full unsubsampled field.")
    return torch.einsum("nk,bnc->bkc", U, fields)

def inverse_graph_fourier_transform(gft_coeffs: torch.Tensor, U: torch.Tensor,):
    """
    Reconstruct spatial fields from graph Fourier coefficients.

    Implements: x^(c) = U hat{x}^(c)

    This is the inverse of graph_fourier_transform. Useful for:
        - Verifying the transform is invertible (when K = N)
        - Visualizing what specific frequency bands look like in space
        - Band-pass filtering fields to isolate scale ranges

    Args:
        gft_coeffs: [B, K, C] — graph Fourier coefficients
        U:          [N, K]    — graph Fourier basis

    Returns:
        [B, N, C] — reconstructed spatial fields
    """
    return torch.einsum("nk,bkc->bnc", U, gft_coeffs)

def estimate_auto_spectra(gft_coeffs: torch.Tensor,):
    """
    Graph power spectral density: the energy distribution per field per frequency.

    hat{p}_c[k] = (1/B) sum_b |hat{X}_{b,k,c}|^2

    Args:
        gft_coeffs: [B, K, C]

    Returns:
        [K, C] — auto_spectra[k, c] = average energy of field c at frequency k
    """
    return (gft_coeffs ** 2).mean(dim=0)
def estimate_cross_spectra(gft_coeffs: torch.Tensor):
    """
    Graph cross-spectral density: the shared structure between field pairs.

    hat{p}_{c1,c2}[k] = (1/B) sum_b hat{X}_{b,k,c1} * hat{X}_{b,k,c2}

    Args:
        gft_coeffs: [B, K, C]

    Returns:
        [K, C, C] — cross_spectra[k, i, j] = shared structure between
                    fields i and j at frequency k.
                    Diagonal entries equal the auto-spectra.
    """
    B = gft_coeffs.shape[0]
    return torch.einsum("bki,bkj->kij", gft_coeffs, gft_coeffs) / B

def compute_coherence(auto_spectra: torch.Tensor, cross_spectra: torch.Tensor, eps: float = 1e-8,):
    """
    Graph coherence: normalized spectral coupling strength.

    c_{XY}[k] = |p_{XY}[k]|^2 / (p_X[k] * p_Y[k] + eps)

    Result is in [0, 1] for each frequency and field pair.
        0 = fields are spectrally independent at this frequency
        1 = fields are perfectly coupled at this frequency

    Args:
        auto_spectra:  [K, C]
        cross_spectra: [K, C, C]
        eps: prevents division by zero

    Returns:
        [K, C, C] — coherence values in [0, 1]
    """
    denom = torch.einsum("ki,kj->kij", auto_spectra, auto_spectra)
    numer = cross_spectra ** 2
    return numer / (denom + eps)



