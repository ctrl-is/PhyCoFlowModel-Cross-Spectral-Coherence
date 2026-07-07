"""
Differentiable cross-spectral objective for direct FFM post-training.
Refactored directly from src.direct_coherence_loss.py
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn

try:
    from .cross_spectral import (
        CrossSpectralConfig,
        compute_band_energy,
        compute_physical_coherence_loss,
        gft,
    )
    from .graph_basis import make_graph_frequency_bands
except ImportError:
    from cross_spectral import (
        CrossSpectralConfig,
        compute_band_energy,
        compute_physical_coherence_loss,
        gft,
    )
    from graph_spectral_coherence.graph_basis import make_graph_frequency_bands


FieldPair = tuple[int, int]


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DirectCrossSpectralConfig:
    """Configuration for direct CSC post-training."""

    enabled: bool = False

    # Loss-component weights inside the physical coherence objective.
    samefreq_weight: float = 1.0
    crossfreq_weight: float = 1.0

    # Optional scale-resolved field-energy objective.
    # Keep this at 0.0 for the first experiment.
    band_energy_weight: float = 0.0
    band_energy_use_log: bool = True

    # None means all unique physical-field pairs.
    field_pairs: Optional[Sequence[FieldPair]] = None

    # Evaluate in physical units when True.
    use_denorm: bool = False

    eps: float = 1e-8

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# Validation helpers
# =============================================================================

def _zero_like_scalar(x: torch.Tensor) -> torch.Tensor:
    """Return a differentiable scalar zero."""
    return x.sum() * 0.0


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(
            f"{name} contains NaN or Inf values."
        )


def _normalize_field_pairs(
    field_pairs: Optional[Sequence[FieldPair]],
    n_fields: int,
) -> Optional[list[FieldPair]]:
    """Validate and deduplicate configured field pairs."""
    if field_pairs is None:
        return None

    normalized = []
    seen = set()

    for raw_pair in field_pairs:
        if len(raw_pair) != 2:
            raise ValueError(
                "Each field pair must contain exactly two indices. "
                f"Received {raw_pair!r}."
            )

        i = int(raw_pair[0])
        j = int(raw_pair[1])

        if i == j:
            raise ValueError(
                f"Field pair {(i, j)} contains the same field twice."
            )

        if not (
            0 <= i < n_fields
            and 0 <= j < n_fields
        ):
            raise ValueError(
                f"Field pair {(i, j)} is outside the valid "
                f"range [0, {n_fields})."
            )

        pair = tuple(sorted((i, j)))

        if pair not in seen:
            normalized.append(pair)
            seen.add(pair)

    if not normalized:
        raise ValueError(
            "field_pairs was supplied but contained no valid pairs."
        )

    return normalized


# =============================================================================
# Existing graph-basis loader
# =============================================================================

def load_cross_spectral_graph_basis(
    graph_basis_path: str | Path,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Load the graph basis already produced by build_graph_basis.py.

    Expected keys:
        U
        eigenvalues
    """
    path = Path(graph_basis_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Graph basis not found: {path}"
        )

    graph_obj = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(graph_obj, dict):
        raise TypeError(
            "Expected the graph-basis file to contain a dictionary."
        )

    if "U" not in graph_obj:
        raise KeyError(
            "Graph-basis file does not contain 'U'."
        )

    if "eigenvalues" not in graph_obj:
        raise KeyError(
            "Graph-basis file does not contain 'eigenvalues'."
        )

    U = torch.as_tensor(
        graph_obj["U"],
        dtype=torch.float32,
    )

    eigenvalues = torch.as_tensor(
        graph_obj["eigenvalues"],
        dtype=torch.float32,
    )

    if U.ndim != 2:
        raise ValueError(
            f"Expected U with shape [N,K], got {tuple(U.shape)}."
        )

    if eigenvalues.ndim != 1:
        raise ValueError(
            "Expected eigenvalues with shape [K], "
            f"got {tuple(eigenvalues.shape)}."
        )

    if U.shape[1] != eigenvalues.shape[0]:
        raise ValueError(
            "Graph mode mismatch: "
            f"U has K={U.shape[1]}, but eigenvalues has "
            f"K={eigenvalues.shape[0]}."
        )

    generated_bands = make_graph_frequency_bands(
        eigenvalues=eigenvalues,
        exclude_zero=True,
        split="thirds",
    )

    bands = {
        str(name): torch.as_tensor(
            indices,
            dtype=torch.long,
        )
        for name, indices in generated_bands.items()
    }

    print(f"[direct-csc] Loaded graph basis: {path}")
    print(f"[direct-csc] U shape: {tuple(U.shape)}")
    print(
        "[direct-csc] Band sizes:",
        {
            name: int(indices.numel())
            for name, indices in bands.items()
        },
    )

    return U, bands


# =============================================================================
# Differentiable band-energy term
# =============================================================================

def _compute_mean_band_energies(
    fields: torch.Tensor,
    U: torch.Tensor,
    bands: dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Compute mean band energy for every field.

    Args:
        fields: [B,N,C]
        U: [N,K]
        bands: graph-frequency band indices

    Returns:
        [M,C], where M is the number of bands.
    """
    coefficients = gft(fields, U)

    energies = []

    for band_name, band_indices in bands.items():
        energy_per_snapshot = compute_band_energy(
            coefficients,
            band_indices,
        )

        # [B,C] -> [C]
        mean_energy = energy_per_snapshot.mean(dim=0)

        energies.append(mean_energy)

    return torch.stack(
        energies,
        dim=0,
    )


def differentiable_band_energy_loss(
    fields_pred: torch.Tensor,
    fields_target: torch.Tensor,
    U: torch.Tensor,
    bands: dict[str, torch.Tensor],
    eps: float = 1e-8,
    use_log: bool = True,
) -> torch.Tensor:
    """
    Compare predicted and target scale-resolved field energies.

    The ratio used in evaluation plots is not optimized directly because
    ratios can become unstable when the target energy is nearly zero.
    """
    energy_pred = _compute_mean_band_energies(
        fields_pred,
        U,
        bands,
    )

    energy_target = _compute_mean_band_energies(
        fields_target,
        U,
        bands,
    )

    if use_log:
        difference = (
            torch.log(energy_pred.clamp_min(eps))
            - torch.log(energy_target.clamp_min(eps))
        )
    else:
        denominator = energy_target.abs().clamp_min(eps)

        difference = (
            energy_pred - energy_target
        ) / denominator

    loss = difference.square().mean()

    _require_finite(
        "band_energy_loss",
        loss,
    )

    return loss


# =============================================================================
# Direct CSC loss
# =============================================================================

class DirectCrossSpectralCoherenceLoss(nn.Module):
    """
    Direct cross-spectral objective for generated terminal fields.

    Both inputs must be complete fields with shape:

        [B,N,C]

    N must equal the node count in the stored graph basis.

    The same-frequency and cross-frequency estimators require multiple
    distinct samples in the batch. A batch size of one is not meaningful.
    """

    def __init__(
        self,
        cfg: DirectCrossSpectralConfig,
        U: torch.Tensor,
        bands: dict[str, torch.Tensor],
    ) -> None:
        super().__init__()

        self.cfg = cfg

        U = torch.as_tensor(
            U,
            dtype=torch.float32,
        )

        if U.ndim != 2:
            raise ValueError(
                f"Expected U with shape [N,K], got {tuple(U.shape)}."
            )

        # The basis is fixed and should not receive gradients.
        # persistent=False avoids copying the large basis into a loss state_dict.
        self.register_buffer(
            "U",
            U,
            persistent=False,
        )

        self._band_names = []

        for position, (name, indices) in enumerate(
            bands.items()
        ):
            indices = torch.as_tensor(
                indices,
                dtype=torch.long,
            )

            if indices.ndim != 1 or indices.numel() == 0:
                raise ValueError(
                    f"Band {name!r} must contain a nonempty "
                    "one-dimensional index tensor."
                )

            self.register_buffer(
                f"_band_indices_{position}",
                indices,
                persistent=False,
            )

            self._band_names.append(str(name))

    def _bands(self) -> dict[str, torch.Tensor]:
        return {
            name: getattr(
                self,
                f"_band_indices_{position}",
            )
            for position, name in enumerate(
                self._band_names
            )
        }

    def _maybe_denormalize(
        self,
        x_gen: torch.Tensor,
        x_ref: torch.Tensor,
        mean: Optional[torch.Tensor],
        std: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.cfg.use_denorm:
            return x_gen, x_ref

        if mean is None or std is None:
            raise ValueError(
                "mean and std are required when use_denorm=True."
            )

        mean = mean.to(
            device=x_gen.device,
            dtype=x_gen.dtype,
        ).view(1, 1, -1)

        std = std.to(
            device=x_gen.device,
            dtype=x_gen.dtype,
        ).view(1, 1, -1)

        _require_finite("mean", mean)
        _require_finite("std", std)

        if torch.any(std <= 0):
            raise ValueError(
                "All field standard deviations must be positive."
            )

        return (
            x_gen * std + mean,
            x_ref * std + mean,
        )

    def forward(
        self,
        x_gen: torch.Tensor,
        x_ref: torch.Tensor,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if x_gen.ndim != 3 or x_ref.ndim != 3:
            raise ValueError(
                "Direct CSC expects [B,N,C] fields, got "
                f"{tuple(x_gen.shape)} and {tuple(x_ref.shape)}."
            )

        if x_gen.shape != x_ref.shape:
            raise ValueError(
                "Generated/reference shape mismatch: "
                f"{tuple(x_gen.shape)} versus {tuple(x_ref.shape)}."
            )

        if x_gen.shape[1] != self.U.shape[0]:
            raise ValueError(
                "The direct CSC objective requires the complete graph. "
                f"Fields have N={x_gen.shape[1]}, while U has "
                f"N={self.U.shape[0]}."
            )

        _require_finite("x_gen", x_gen)
        _require_finite("x_ref", x_ref)

        zero = _zero_like_scalar(x_gen)

        if not self.cfg.enabled:
            return zero, {
                "samefreq_loss": zero,
                "crossfreq_loss": zero,
                "band_energy_loss": zero,
                "total_loss": zero,
            }

        # Your spectral estimators compare statistics across the batch.
        if x_gen.shape[0] < 2:
            raise ValueError(
                "Direct cross-spectral training requires batch_size >= 2. "
                "With B=1, centered cross-band energies are zero and "
                "same-frequency coherence becomes degenerate."
            )

        x_gen, x_ref = self._maybe_denormalize(
            x_gen,
            x_ref,
            mean,
            std,
        )

        U = self.U.to(
            dtype=x_gen.dtype
        )

        bands = self._bands()

        field_pairs = _normalize_field_pairs(
            self.cfg.field_pairs,
            n_fields=x_gen.shape[-1],
        )

        repo_cfg = CrossSpectralConfig(
            eps=float(self.cfg.eps),
            eta_crossfreq=1.0,
            field_pairs=field_pairs,
        )

        outputs = compute_physical_coherence_loss(
            fields_pred=x_gen,
            fields_target=x_ref,
            U=U,
            bands=bands,
            cfg=repo_cfg,
        )

        samefreq_loss = outputs["L_same"]
        crossfreq_loss = outputs["L_crossfreq"]

        _require_finite(
            "samefreq_loss",
            samefreq_loss,
        )

        _require_finite(
            "crossfreq_loss",
            crossfreq_loss,
        )

        band_energy_loss = zero

        if float(self.cfg.band_energy_weight) != 0.0:
            # Recompute instead of using outputs["auto_spectra_pred"],
            # because cross_spectral.py detaches those diagnostics.
            band_energy_loss = differentiable_band_energy_loss(
                fields_pred=x_gen,
                fields_target=x_ref,
                U=U,
                bands=bands,
                eps=float(self.cfg.eps),
                use_log=bool(
                    self.cfg.band_energy_use_log
                ),
            )

        total_loss = (
            float(self.cfg.samefreq_weight)
            * samefreq_loss
            + float(self.cfg.crossfreq_weight)
            * crossfreq_loss
            + float(self.cfg.band_energy_weight)
            * band_energy_loss
        )

        _require_finite(
            "total_loss",
            total_loss,
        )

        return total_loss, {
            "samefreq_loss": samefreq_loss,
            "crossfreq_loss": crossfreq_loss,
            "band_energy_loss": band_energy_loss,
            "total_loss": total_loss,
        }


# =============================================================================
# Constructor
# =============================================================================

def build_direct_cross_spectral_loss(
    cfg: DirectCrossSpectralConfig,
    graph_basis_path: str | Path,
    device: torch.device | str,
) -> DirectCrossSpectralCoherenceLoss:
    U, bands = load_cross_spectral_graph_basis(
        graph_basis_path
    )

    module = DirectCrossSpectralCoherenceLoss(
        cfg=cfg,
        U=U,
        bands=bands,
    )

    return module.to(device)


# =============================================================================
# Gradient test
# =============================================================================

def gradient_sanity_check(
    device: str | torch.device = "cpu",
) -> bool:
    """Check that the CSC objective produces a finite nonzero gradient."""
    device = torch.device(device)

    batch_size = 4
    n_points = 24
    n_modes = 12
    n_fields = 4

    x_gen = torch.randn(
        batch_size,
        n_points,
        n_fields,
        device=device,
        requires_grad=True,
    )

    x_ref = torch.randn(
        batch_size,
        n_points,
        n_fields,
        device=device,
    )

    random_basis = torch.randn(
        n_points,
        n_modes,
        device=device,
    )

    U, _ = torch.linalg.qr(
        random_basis,
        mode="reduced",
    )

    bands = {
        "low": torch.arange(1, 4, device=device),
        "mid": torch.arange(4, 8, device=device),
        "high": torch.arange(8, 12, device=device),
    }

    cfg = DirectCrossSpectralConfig(
        enabled=True,
        samefreq_weight=1.0,
        crossfreq_weight=1.0,
        band_energy_weight=0.0,
    )

    loss_module = DirectCrossSpectralCoherenceLoss(
        cfg=cfg,
        U=U,
        bands=bands,
    ).to(device)

    loss, components = loss_module(
        x_gen=x_gen,
        x_ref=x_ref,
    )

    loss.backward()

    valid = bool(
        x_gen.grad is not None
        and torch.isfinite(x_gen.grad).all()
        and torch.linalg.vector_norm(x_gen.grad).item() > 0.0
    )

    print(
        "samefreq_loss:",
        float(
            components["samefreq_loss"]
            .detach()
            .cpu()
        ),
    )

    print(
        "crossfreq_loss:",
        float(
            components["crossfreq_loss"]
            .detach()
            .cpu()
        ),
    )

    print(
        "total_loss:",
        float(
            components["total_loss"]
            .detach()
            .cpu()
        ),
    )

    print(
        "gradient_norm:",
        (
            float(
                torch.linalg.vector_norm(x_gen.grad)
                .detach()
                .cpu()
            )
            if x_gen.grad is not None
            else 0.0
        ),
    )

    return valid


__all__ = [
    "DirectCrossSpectralConfig",
    "DirectCrossSpectralCoherenceLoss",
    "build_direct_cross_spectral_loss",
    "differentiable_band_energy_loss",
    "gradient_sanity_check",
    "load_cross_spectral_graph_basis",
]