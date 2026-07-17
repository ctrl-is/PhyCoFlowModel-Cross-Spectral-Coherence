"""Differentiable direct cross-spectral coherence utilities for PointCloudFFM post-training.

This module preserves the original direct post-training pipeline:
- point selection helpers
- differentiable RF rollout
- observation-consistency handling
- weighted-sum / ConFIG optimizer updates
- gradient diagnostics

Only the physical-coherence objective is replaced with graph cross-spectral
coherence.

Sidenote: must change train_pointcloud_ffm.py to use this
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn

from cross_spectral.cross_spectral import (
    CrossSpectralConfig,
    compute_band_energy,
    compute_physical_coherence_loss,
    gft,
)
from cross_spectral.graph import make_graph_frequency_bands
from obs_consistency import (
    apply_endpoint_observation_consistency,
    build_pointwise_observation_maps,
    build_smooth_observation_maps,
    normalize_obs_consistency_mode,
    scatter_observed_values,
)


FieldPair = tuple[int, int]


@dataclass
class DirectCrossSpectralConfig:
    """Configuration for differentiable direct CSC post-training."""

    enabled: bool = False
    samefreq_weight: float = 1.0
    crossfreq_weight: float = 1.0

    # Optional scale-resolved energy term.
    band_energy_weight: float = 0.0
    band_energy_use_log: bool = True

    # None means all unique physical-field pairs.
    field_pairs: Optional[Sequence[FieldPair]] = None

    # Evaluate in physical units when True.
    use_denorm: bool = False
    eps: float = 1.0e-8

    def to_dict(self) -> dict:
        return asdict(self)


def _zero_like_scalar(x: torch.Tensor) -> torch.Tensor:
    """Return a scalar zero that remains connected to autograd."""
    return x.sum() * 0.0


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains NaN or Inf values.")


def _normalize_field_pairs(
    field_pairs: Optional[Sequence[FieldPair]],
    n_fields: int,
) -> Optional[list[FieldPair]]:
    """Validate, canonicalize, and deduplicate configured field pairs."""
    if field_pairs is None:
        return None

    normalized: list[FieldPair] = []
    seen: set[FieldPair] = set()

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

        if not (0 <= i < n_fields and 0 <= j < n_fields):
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


def load_cross_spectral_graph_basis(
    graph_basis_path: str | Path,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Load and validate a graph basis produced by build_graph_basis.py."""
    path = Path(graph_basis_path)

    if not path.exists():
        raise FileNotFoundError(f"Graph basis not found: {path}")

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
        raise KeyError("Graph-basis file does not contain 'U'.")
    if "eigenvalues" not in graph_obj:
        raise KeyError(
            "Graph-basis file does not contain 'eigenvalues'."
        )

    U = torch.as_tensor(graph_obj["U"], dtype=torch.float32)
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
        str(name): torch.as_tensor(indices, dtype=torch.long)
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


def _compute_mean_band_energies(
    fields: torch.Tensor,
    U: torch.Tensor,
    bands: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Return mean energy per graph-frequency band and physical field."""
    coefficients = gft(fields, U)
    energies = []

    for band_indices in bands.values():
        energy_per_snapshot = compute_band_energy(
            coefficients,
            band_indices,
        )
        energies.append(energy_per_snapshot.mean(dim=0))

    return torch.stack(energies, dim=0)


def differentiable_band_energy_loss(
    fields_pred: torch.Tensor,
    fields_target: torch.Tensor,
    U: torch.Tensor,
    bands: dict[str, torch.Tensor],
    eps: float = 1.0e-8,
    use_log: bool = True,
) -> torch.Tensor:
    """Compare predicted and target scale-resolved field energies."""
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
        difference = (energy_pred - energy_target) / denominator

    loss = difference.square().mean()
    _require_finite("band_energy_loss", loss)
    return loss


class DirectCrossSpectralCoherenceLoss(nn.Module):
    """Differentiable CSC objective for complete generated fields [B,N,C]."""

    def __init__(
        self,
        cfg: DirectCrossSpectralConfig,
        U: torch.Tensor,
        bands: dict[str, torch.Tensor],
    ) -> None:
        super().__init__()
        self.cfg = cfg

        U = torch.as_tensor(U, dtype=torch.float32)
        if U.ndim != 2:
            raise ValueError(
                f"Expected U with shape [N,K], got {tuple(U.shape)}."
            )

        self.register_buffer("U", U, persistent=False)
        self._band_names: list[str] = []

        for position, (name, indices) in enumerate(bands.items()):
            indices = torch.as_tensor(indices, dtype=torch.long)
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
            name: getattr(self, f"_band_indices_{position}")
            for position, name in enumerate(self._band_names)
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

        return x_gen * std + mean, x_ref * std + mean

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
                "Direct CSC requires the complete graph. "
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

        if x_gen.shape[0] < 2:
            raise ValueError(
                "Direct CSC training requires batch_size >= 2."
            )

        x_gen, x_ref = self._maybe_denormalize(
            x_gen,
            x_ref,
            mean,
            std,
        )

        U = self.U.to(dtype=x_gen.dtype)
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

        _require_finite("samefreq_loss", samefreq_loss)
        _require_finite("crossfreq_loss", crossfreq_loss)

        band_energy_loss = zero
        if float(self.cfg.band_energy_weight) != 0.0:
            band_energy_loss = differentiable_band_energy_loss(
                fields_pred=x_gen,
                fields_target=x_ref,
                U=U,
                bands=bands,
                eps=float(self.cfg.eps),
                use_log=bool(self.cfg.band_energy_use_log),
            )

        total_loss = (
            float(self.cfg.samefreq_weight) * samefreq_loss
            + float(self.cfg.crossfreq_weight) * crossfreq_loss
            + float(self.cfg.band_energy_weight) * band_energy_loss
        )
        _require_finite("total_loss", total_loss)

        return total_loss, {
            "samefreq_loss": samefreq_loss,
            "crossfreq_loss": crossfreq_loss,
            "band_energy_loss": band_energy_loss,
            "total_loss": total_loss,
        }


def build_direct_cross_spectral_loss(
    cfg: DirectCrossSpectralConfig,
    graph_basis_path: str | Path,
    device: torch.device | str,
) -> DirectCrossSpectralCoherenceLoss:
    """Build the CSC loss while preserving the original training utilities."""
    U, bands = load_cross_spectral_graph_basis(graph_basis_path)
    module = DirectCrossSpectralCoherenceLoss(
        cfg=cfg,
        U=U,
        bands=bands,
    )
    return module.to(device)


def sample_coherence_points(
    coords: torch.Tensor,
    fields: torch.Tensor,
    n_points: Optional[int],
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Uniformly subsample empirical field points for coherence evaluation."""
    if coords.ndim != 3 or fields.ndim != 3:
        raise ValueError(f"coords and fields must be [B, N, *], got {tuple(coords.shape)} and {tuple(fields.shape)}")
    if coords.shape[:2] != fields.shape[:2]:
        raise ValueError(f"coords/fields point shape mismatch: {tuple(coords.shape)} vs {tuple(fields.shape)}")
    if n_points is None or int(n_points) >= coords.shape[1]:
        return coords, fields, None

    bsz, n_total, coord_dim = coords.shape
    n_select = max(1, int(n_points))
    indices = []
    for _ in range(bsz):
        indices.append(torch.randperm(n_total, device=coords.device, generator=generator)[:n_select].sort().values)
    idx = torch.stack(indices, dim=0)
    coord_idx = idx.unsqueeze(-1).expand(-1, -1, coord_dim)
    field_idx = idx.unsqueeze(-1).expand(-1, -1, fields.shape[-1])
    return torch.gather(coords, 1, coord_idx), torch.gather(fields, 1, field_idx), idx


def _call_velocity_model(
    ffm_model: nn.Module,
    t: torch.Tensor,
    x_t: torch.Tensor,
    coords: torch.Tensor,
    obs_coords: torch.Tensor,
    obs_values: torch.Tensor,
    obs_mask: torch.Tensor,
    obs_field_ids: torch.Tensor,
    obs_indices: Optional[torch.Tensor],
) -> torch.Tensor:
    kwargs = {
        "t": t,
        "x_t": x_t,
        "coords": coords,
        "obs_coords": obs_coords,
        "obs_values": obs_values,
        "obs_mask": obs_mask,
        "obs_field_ids": obs_field_ids,
    }
    if getattr(ffm_model, "requires_full_grid", False):
        if obs_indices is None:
            raise ValueError("This FFM model requires full-grid obs_indices for velocity calls.")
        kwargs["obs_indices"] = obs_indices
    return ffm_model.model(**kwargs)


def _validate_pointwise_indices(
    coords: torch.Tensor,
    obs_indices: Optional[torch.Tensor],
    mode: str,
) -> None:
    if mode not in ("default_hard", "endpoint"):
        return
    if obs_indices is None:
        raise ValueError(f"obs_consistency_mode={mode!r} requires obs_indices.")
    valid = obs_indices >= 0
    if torch.any(valid & (obs_indices >= coords.shape[1])):
        raise ValueError(
            f"obs_consistency_mode={mode!r} requires obs_indices valid for the rollout coordinate set. "
            "Use endpoint_smooth for point-subset coherence rollouts."
        )


def differentiable_rf_rollout(
    ffm_model: nn.Module,
    coords: torch.Tensor,
    obs_coords: torch.Tensor,
    obs_values: torch.Tensor,
    obs_mask: torch.Tensor,
    obs_field_ids: torch.Tensor,
    obs_indices: Optional[torch.Tensor],
    n_steps: int = 2,
    ode_solver: str = "euler",
    obs_consistency_mode: str = "endpoint_smooth",
    obs_consistency_strength: float = 1.0,
    obs_consistency_sigma: float = 0.05,
    obs_consistency_schedule_power: float = 2.0,
    obs_consistency_final_clamp: bool = True,
) -> torch.Tensor:
    """Differentiable terminal rollout for direct coherence training."""
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if ode_solver not in ("euler", "heun"):
        raise ValueError(f"Unsupported ode_solver={ode_solver!r}; expected 'euler' or 'heun'.")
    if getattr(ffm_model, "requires_full_grid", False):
        if obs_indices is None:
            raise ValueError("Full-grid FFM rollouts require obs_indices.")
        if coords.shape[1] != int(getattr(ffm_model.model, "Num_x", 0)) * int(getattr(ffm_model.model, "Num_y", 0)):
            raise ValueError("Full-grid FFM rollouts require full-grid coordinates.")

    mode = normalize_obs_consistency_mode(obs_consistency_mode)
    _validate_pointwise_indices(coords, obs_indices, mode)

    bsz = coords.shape[0]
    x = ffm_model.sample_source(coords)
    value_map = None
    mask_map = None
    if mode == "endpoint":
        value_map, mask_map = build_pointwise_observation_maps(
            coords=coords,
            obs_values=obs_values,
            obs_mask=obs_mask,
            obs_indices=obs_indices,
            obs_field_ids=obs_field_ids,
            n_fields=ffm_model.model.n_fields,
        )
    elif mode == "endpoint_smooth":
        value_map, mask_map = build_smooth_observation_maps(
            coords=coords,
            obs_coords=obs_coords,
            obs_values=obs_values,
            obs_mask=obs_mask,
            obs_field_ids=obs_field_ids,
            n_fields=ffm_model.model.n_fields,
            sigma=float(obs_consistency_sigma),
        )

    ts = torch.linspace(0.0, 1.0, int(n_steps) + 1, device=coords.device, dtype=coords.dtype)
    for i in range(int(n_steps)):
        t0 = ts[i].expand(bsz)
        dt = ts[i + 1] - ts[i]
        v0 = _call_velocity_model(
            ffm_model, t0, x, coords, obs_coords, obs_values, obs_mask, obs_field_ids, obs_indices
        )
        if mode in ("endpoint", "endpoint_smooth"):
            v0 = apply_endpoint_observation_consistency(
                x_t=x,
                v=v0,
                t=t0,
                value_map=value_map,
                mask_map=mask_map,
                strength=obs_consistency_strength,
                schedule_power=obs_consistency_schedule_power,
            )

        if ode_solver == "heun":
            x_euler = x + dt * v0
            t1 = ts[i + 1].expand(bsz)
            v1 = _call_velocity_model(
                ffm_model, t1, x_euler, coords, obs_coords, obs_values, obs_mask, obs_field_ids, obs_indices
            )
            if mode in ("endpoint", "endpoint_smooth") and float(ts[i + 1].item()) < 1.0:
                v1 = apply_endpoint_observation_consistency(
                    x_t=x_euler,
                    v=v1,
                    t=t1,
                    value_map=value_map,
                    mask_map=mask_map,
                    strength=obs_consistency_strength,
                    schedule_power=obs_consistency_schedule_power,
                )
            x = x + 0.5 * dt * (v0 + v1)
        else:
            x = x + dt * v0

        if mode == "default_hard":
            x = scatter_observed_values(x, obs_values, obs_mask, obs_indices, obs_field_ids, strength=1.0)

    if obs_consistency_final_clamp and mode != "none" and obs_indices is not None:
        valid = obs_indices >= 0
        indices_fit_current_coords = not torch.any(valid & (obs_indices >= coords.shape[1]))
        if indices_fit_current_coords:
            x = scatter_observed_values(x, obs_values, obs_mask, obs_indices, obs_field_ids, strength=1.0)
    return x


def _gradient_vector_from_model(model: nn.Module) -> torch.Tensor:
    pieces = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            pieces.append(torch.zeros_like(param).reshape(-1))
        else:
            pieces.append(param.grad.reshape(-1))
    if not pieces:
        raise RuntimeError("No trainable parameters found for gradient diagnostics.")
    return torch.cat(pieces)


def _weighted_sum_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loss: torch.Tensor,
    coherence_loss: torch.Tensor,
    data_weight: float,
    coherence_weight: float,
    grad_clip_norm: Optional[float],
) -> dict:
    optimizer.zero_grad(set_to_none=True)
    total_loss = float(data_weight) * data_loss + float(coherence_weight) * coherence_loss
    total_loss.backward()
    grad_vec = _gradient_vector_from_model(model)
    if not torch.isfinite(grad_vec).all():
        raise FloatingPointError("Weighted-sum gradient contains NaN or Inf values.")
    if grad_clip_norm is not None and float(grad_clip_norm) > 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
    optimizer.step()
    norm = torch.linalg.vector_norm(grad_vec.detach())
    return {
        "data_grad_norm": float("nan"),
        "coherence_grad_norm": float("nan"),
        "gradient_cosine": float("nan"),
        "gradient_conflict": False,
        "config_fallback_used": False,
        "combined_grad_norm": float(norm.detach().cpu()),
    }


def apply_two_objective_update(
    model,
    optimizer,
    data_loss,
    coherence_loss,
    mode,
    data_weight,
    coherence_weight,
    grad_clip_norm,
    config_missing_behavior="error",
) -> dict:
    """Apply weighted-sum or ConFIG two-objective update.

    In ``config`` mode, ``data_weight`` and ``coherence_weight`` are optional
    gradient pre-scales before the conflict-free update. Manual relative tuning
    should usually be done with ``weighted_sum`` mode.
    """
    mode = str(mode or "weighted_sum").strip().lower()
    if mode == "weighted_sum":
        return _weighted_sum_update(
            model, optimizer, data_loss, coherence_loss, data_weight, coherence_weight, grad_clip_norm
        )
    if mode != "config":
        raise ValueError("gradient_balance_mode must be 'weighted_sum' or 'config'.")

    try:
        from conflictfree.grad_operator import ConFIG_update
        from conflictfree.utils import apply_gradient_vector, get_gradient_vector
    except ImportError as exc:
        if str(config_missing_behavior) == "weighted_sum":
            warnings.warn(
                "conflictfree is not installed; falling back to weighted_sum. "
                "Install with: pip install conflictfree",
                RuntimeWarning,
                stacklevel=2,
            )
            return _weighted_sum_update(
                model, optimizer, data_loss, coherence_loss, data_weight, coherence_weight, grad_clip_norm
            )
        raise ImportError(
            "gradient_balance_mode='config' requires the optional conflictfree package. "
            "Install with: pip install conflictfree"
        ) from exc

    optimizer.zero_grad(set_to_none=True)
    (float(data_weight) * data_loss).backward()
    g_data = get_gradient_vector(model)
    optimizer.zero_grad(set_to_none=True)
    (float(coherence_weight) * coherence_loss).backward()
    g_coherence = get_gradient_vector(model)

    data_ok = torch.isfinite(g_data).all()
    coherence_ok = torch.isfinite(g_coherence).all()
    data_norm = torch.linalg.vector_norm(g_data.detach())
    coherence_norm = torch.linalg.vector_norm(g_coherence.detach())
    denom = (data_norm * coherence_norm).clamp_min(1e-12)
    cosine = torch.dot(g_data.detach().reshape(-1), g_coherence.detach().reshape(-1)) / denom

    optimizer.zero_grad(set_to_none=True)
    fallback = False
    if not data_ok:
        raise FloatingPointError("Data gradient contains NaN or Inf values.")
    if (not coherence_ok) or float(coherence_norm.detach().cpu()) == 0.0:
        warnings.warn("Invalid or zero coherence gradient; falling back to data gradient.", RuntimeWarning, stacklevel=2)
        apply_gradient_vector(model, g_data)
        fallback = True
    else:
        g_config = ConFIG_update([g_data, g_coherence])
        if not torch.isfinite(g_config).all():
            warnings.warn("ConFIG gradient was invalid; falling back to data gradient.", RuntimeWarning, stacklevel=2)
            apply_gradient_vector(model, g_data)
            fallback = True
        else:
            apply_gradient_vector(model, g_config)

    if grad_clip_norm is not None and float(grad_clip_norm) > 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
    optimizer.step()
    return {
        "data_grad_norm": float(data_norm.detach().cpu()),
        "coherence_grad_norm": float(coherence_norm.detach().cpu()),
        "gradient_cosine": float(cosine.detach().cpu()),
        "gradient_conflict": bool(float(cosine.detach().cpu()) < 0.0),
        "config_fallback_used": fallback,
    }


def gradient_sanity_check(
    graph_basis_path: str | Path,
    device: str | torch.device = "cpu",
) -> bool:
    """Check that the CSC objective backpropagates to generated fields."""
    device = torch.device(device)
    U, bands = load_cross_spectral_graph_basis(graph_basis_path)
    U = U.to(device)

    x_gen = torch.randn(
        4,
        U.shape[0],
        5,
        device=device,
        requires_grad=True,
    )
    x_ref = torch.randn(
        4,
        U.shape[0],
        5,
        device=device,
    )

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

    loss, _ = loss_module(x_gen=x_gen, x_ref=x_ref)
    loss.backward()

    return bool(
        x_gen.grad is not None
        and torch.isfinite(x_gen.grad).all()
        and torch.linalg.vector_norm(x_gen.grad).item() > 0.0
    )


__all__ = [
    "DirectCrossSpectralConfig",
    "DirectCrossSpectralCoherenceLoss",
    "apply_two_objective_update",
    "build_direct_cross_spectral_loss",
    "differentiable_band_energy_loss",
    "differentiable_rf_rollout",
    "gradient_sanity_check",
    "load_cross_spectral_graph_basis",
    "sample_coherence_points",
]