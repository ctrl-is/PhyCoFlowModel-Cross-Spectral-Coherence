from __future__ import annotations

import json
import sys
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from coherence_dist import per_channel_w2  # noqa: E402
from helpers import TurbulentCombustionH5Dataset  # noqa: E402

from cross_spectral import (  # noqa: E402
    CrossSpectralConfig,
    compute_physical_coherence_loss,
)
from direct_cross_spectral_loss import (  # noqa: E402
    load_cross_spectral_graph_basis,
)


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

DATA_PATH = PROJECT_ROOT.parent / "Dataset/Merged_CH4COTU1P.h5"

STATS_PATH = (
    PROJECT_ROOT
    / "Save_TrainedModel"
    / "ffm_tc_pointcloud_DemoN30_20260610_173403"
    / "dataset_stats.pt"
)

GRAPH_BASIS_PATH = (
    PROJECT_ROOT
    / "Save_Graph"
    / "graph_basis_k16_modes384.pt"
)

OUTPUT_DIR = PROJECT_ROOT / "Evaluation" / "controlled_corruptions"

FIELD_NAMES = ("CH4", "CO", "T", "U_1", "p")
FIELD_INDICES = {name: index for index, name in enumerate(FIELD_NAMES)}

CH4_FIELD_INDEX = FIELD_INDICES["CH4"]
CO_FIELD_INDEX = FIELD_INDICES["CO"]
T_FIELD_INDEX = FIELD_INDICES["T"]
U1_FIELD_INDEX = FIELD_INDICES["U_1"]
P_FIELD_INDEX = FIELD_INDICES["p"]

GRID_NUM_X = 403
GRID_NUM_Y = 100
VIS_SNAPSHOT_INDEX = 0

NUM_SNAPSHOTS = 24
SEED = 42
EPS = 1.0e-8

# Test 1: use this fraction of the largest L2 target that both corruption
# families can attain. The resulting L2 values are analytically matched.
TEST1_MATCHED_L2_FRACTION = 0.75

# Test 3: candidate whole-snapshot permutations used to find two cases with
# closely matched marginal self-error but substantially different L2 error.
TEST3_NUM_CANDIDATES = 512
TEST3_SELF_RELATIVE_TOLERANCE = 0.05
TEST3_MIN_L2_GAP = 0.05

# Representative field pair for cross-band coupling heatmaps.
HEATMAP_FIELD_PAIR = (CO_FIELD_INDEX, T_FIELD_INDEX)


# ---------------------------------------------------------------------------
# Small data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Test3Candidate:
    permutation: torch.Tensor
    relative_l2_mean: float
    self_marginal_error: float


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def load_ground_truth_batch(
    num_snapshots: int,
) -> tuple[
    TurbulentCombustionH5Dataset,
    torch.Tensor,
    torch.Tensor,
]:
    """Load a deterministic batch of normalized ground-truth test fields."""
    require_file(DATA_PATH, "Dataset")
    require_file(STATS_PATH, "Dataset statistics")

    dataset = TurbulentCombustionH5Dataset(
        h5_path=str(DATA_PATH),
        split="test",
        train_ratio=0.9,
        seed=SEED,
        field_names=FIELD_NAMES,
        stats_path=str(STATS_PATH),
        time_stride=1,
    )

    if num_snapshots < 2:
        raise ValueError(
            "Cross-frequency coupling requires at least two snapshots."
        )

    if num_snapshots > len(dataset):
        raise ValueError(
            f"Requested {num_snapshots} snapshots, but the test split "
            f"contains only {len(dataset)}."
        )

    samples = [dataset[index] for index in range(num_snapshots)]

    fields_true = torch.stack(
        [sample["fields"] for sample in samples],
        dim=0,
    )

    time_indices = torch.stack(
        [sample["time_index"] for sample in samples],
        dim=0,
    )

    return dataset, fields_true, time_indices


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------


def relative_l2_per_field(
    fields_corrupted: torch.Tensor,
    fields_reference: torch.Tensor,
    eps: float = EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute global relative L2 separately for every physical field."""
    if fields_corrupted.shape != fields_reference.shape:
        raise ValueError(
            "Corrupted and reference fields must have equal shapes. "
            f"Received {tuple(fields_corrupted.shape)} and "
            f"{tuple(fields_reference.shape)}."
        )

    difference = fields_corrupted - fields_reference

    numerator = difference.square().sum(dim=(0, 1)).sqrt()
    denominator = (
        fields_reference.square()
        .sum(dim=(0, 1))
        .sqrt()
        .clamp_min(eps)
    )

    per_field_relative_l2 = numerator / denominator
    mean_relative_l2 = per_field_relative_l2.mean()

    return mean_relative_l2, per_field_relative_l2


def marginal_self_error(
    fields_corrupted: torch.Tensor,
    fields_reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the marginal-distribution self-consistency error.

    This exactly matches the self-only behavior of DirectGlobalCoherenceLoss:
    per-channel empirical 1-D W2^2 is computed independently for each
    snapshot, then averaged across snapshots and fields.
    """
    if fields_corrupted.shape != fields_reference.shape:
        raise ValueError(
            "Corrupted and reference fields must have equal shapes."
        )

    per_snapshot = torch.stack(
        [
            per_channel_w2(fields_corrupted[b], fields_reference[b])
            for b in range(fields_corrupted.shape[0])
        ],
        dim=0,
    )
    per_field = per_snapshot.mean(dim=0)
    return per_field.mean(), per_field


def pairwise_crossfreq_errors(
    q_corrupted: torch.Tensor,
    q_reference: torch.Tensor,
) -> dict[str, float]:
    """Return each field pair's contribution to L_crossfreq."""
    if q_corrupted.shape != q_reference.shape:
        raise ValueError("Cross-band coupling tensors must have equal shapes.")

    num_bands, _, num_fields, _ = q_corrupted.shape
    off_diagonal = ~torch.eye(
        num_bands,
        dtype=torch.bool,
        device=q_corrupted.device,
    )

    results: dict[str, float] = {}
    for field_i in range(num_fields):
        for field_j in range(field_i + 1, num_fields):
            difference = (
                q_corrupted[:, :, field_i, field_j]
                - q_reference[:, :, field_i, field_j]
            )
            pair_error = difference[off_diagonal].abs().square().mean()
            pair_name = f"{FIELD_NAMES[field_i]}-{FIELD_NAMES[field_j]}"
            results[pair_name] = float(pair_error.detach().cpu())

    return results


@torch.no_grad()
def evaluate_corruption(
    fields_corrupted: torch.Tensor,
    fields_reference: torch.Tensor,
    graph_basis: torch.Tensor,
    frequency_bands: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Evaluate one corrupted batch against its ground-truth reference."""
    if fields_corrupted.ndim != 3:
        raise ValueError(
            "Expected corrupted fields with shape [B, N, C], "
            f"received {tuple(fields_corrupted.shape)}."
        )

    if fields_corrupted.shape != fields_reference.shape:
        raise ValueError(
            "Corrupted/reference shape mismatch: "
            f"{tuple(fields_corrupted.shape)} versus "
            f"{tuple(fields_reference.shape)}."
        )

    if fields_corrupted.shape[1] != graph_basis.shape[0]:
        raise ValueError(
            "Spatial node count does not match graph basis: "
            f"fields contain N={fields_corrupted.shape[1]}, "
            f"but U contains N={graph_basis.shape[0]}."
        )

    mean_l2, per_field_l2 = relative_l2_per_field(
        fields_corrupted,
        fields_reference,
    )

    self_error, self_per_field = marginal_self_error(
        fields_corrupted,
        fields_reference,
    )

    spectral_config = CrossSpectralConfig(
        eps=EPS,
        eta_crossfreq=1.0,
        field_pairs=None,
    )

    spectral_outputs = compute_physical_coherence_loss(
        fields_pred=fields_corrupted,
        fields_target=fields_reference,
        U=graph_basis,
        bands=frequency_bands,
        cfg=spectral_config,
    )

    q_corrupted = spectral_outputs["Q_pred"]
    q_reference = spectral_outputs["Q_target"]

    return {
        "relative_l2_mean": float(mean_l2.detach().cpu()),
        "relative_l2_per_field": {
            field_name: float(value)
            for field_name, value in zip(
                FIELD_NAMES,
                per_field_l2.detach().cpu(),
            )
        },
        "self_marginal_error": float(self_error.detach().cpu()),
        "self_marginal_error_per_field": {
            field_name: float(value)
            for field_name, value in zip(
                FIELD_NAMES,
                self_per_field.detach().cpu(),
            )
        },
        "crossfreq_mutual_error": float(
            spectral_outputs["L_crossfreq"].detach().cpu()
        ),
        "samefreq_diagnostic": float(
            spectral_outputs["L_same"].detach().cpu()
        ),
        "crossfreq_pair_errors": pairwise_crossfreq_errors(
            q_corrupted,
            q_reference,
        ),
        "band_names": list(spectral_outputs["band_names"]),
        "Q_corrupted": q_corrupted.detach().cpu().tolist(),
        "Q_reference": q_reference.detach().cpu().tolist(),
    }


# ---------------------------------------------------------------------------
# Corruption constructors
# ---------------------------------------------------------------------------


def make_cyclic_donor_permutation(
    batch_size: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Create a deterministic, fixed-point-free permutation."""
    if batch_size < 2:
        raise ValueError("A donor permutation requires at least two snapshots.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    shift = int(
        torch.randint(
            low=1,
            high=batch_size,
            size=(1,),
            generator=generator,
        ).item()
    )
    return torch.roll(
        torch.arange(batch_size, device=device),
        shifts=shift,
    )


def mix_all_fields_with_donors(
    fields: torch.Tensor,
    donor_permutation: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    """Mix every field with the same donor snapshot."""
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must lie in [0, 1].")
    donor_fields = fields[donor_permutation]
    return (1.0 - strength) * fields + strength * donor_fields


def mix_single_field_with_donors(
    fields: torch.Tensor,
    donor_permutation: torch.Tensor,
    field_index: int,
    strength: float,
) -> torch.Tensor:
    """Mix only one physical field with a donor snapshot."""
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must lie in [0, 1].")

    corrupted = fields.clone()
    donor_values = fields[donor_permutation, :, field_index]
    corrupted[:, :, field_index] = (
        (1.0 - strength) * fields[:, :, field_index]
        + strength * donor_values
    )
    return corrupted


def build_matched_l2_corruptions(
    fields: torch.Tensor,
    donor_permutation: torch.Tensor,
    field_index: int,
    target_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    Construct Test 1 corruptions with analytically matched mean relative L2.

    Relative L2 scales linearly with the mixing strength because each
    corruption is an affine interpolation between fixed endpoint fields.
    """
    if not 0.0 < target_fraction <= 1.0:
        raise ValueError("target_fraction must lie in (0, 1].")

    common_endpoint = mix_all_fields_with_donors(
        fields,
        donor_permutation,
        strength=1.0,
    )
    single_endpoint = mix_single_field_with_donors(
        fields,
        donor_permutation,
        field_index=field_index,
        strength=1.0,
    )

    common_endpoint_l2, _ = relative_l2_per_field(common_endpoint, fields)
    single_endpoint_l2, _ = relative_l2_per_field(single_endpoint, fields)

    common_endpoint_value = float(common_endpoint_l2.detach().cpu())
    single_endpoint_value = float(single_endpoint_l2.detach().cpu())

    attainable_target = target_fraction * min(
        common_endpoint_value,
        single_endpoint_value,
    )

    if attainable_target <= 0.0:
        raise RuntimeError("Could not construct a nonzero matched-L2 target.")

    common_strength = attainable_target / common_endpoint_value
    single_strength = attainable_target / single_endpoint_value

    common_corruption = mix_all_fields_with_donors(
        fields,
        donor_permutation,
        strength=common_strength,
    )
    single_corruption = mix_single_field_with_donors(
        fields,
        donor_permutation,
        field_index=field_index,
        strength=single_strength,
    )

    metadata = {
        "target_relative_l2": attainable_target,
        "common_mix_strength": common_strength,
        "single_field_mix_strength": single_strength,
        "common_endpoint_relative_l2": common_endpoint_value,
        "single_field_endpoint_relative_l2": single_endpoint_value,
    }
    return common_corruption, single_corruption, metadata


def spatially_shuffle_single_field(
    fields: torch.Tensor,
    field_index: int,
    seed: int,
) -> torch.Tensor:
    """
    Randomly permute one field's spatial locations within every snapshot.

    The marginal distribution is preserved exactly, but spatial and
    cross-field relationships are disrupted.
    """
    if fields.ndim != 3:
        raise ValueError(
            f"Expected fields with shape [B, N, C], "
            f"received {tuple(fields.shape)}."
        )

    batch_size, num_nodes, num_fields = fields.shape

    if not 0 <= field_index < num_fields:
        raise ValueError(
            f"field_index must be between 0 and {num_fields - 1}, "
            f"received {field_index}."
        )

    corrupted = fields.clone()
    generator = torch.Generator(device=fields.device)
    generator.manual_seed(seed)

    for batch_index in range(batch_size):
        permutation = torch.randperm(
            num_nodes,
            generator=generator,
            device=fields.device,
        )
        corrupted[batch_index, :, field_index] = fields[
            batch_index,
            permutation,
            field_index,
        ]

    return corrupted


def precompute_snapshot_pair_costs(
    fields: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Precompute exact costs for Test 3 permutation selection.

    Returns
    -------
    squared_l2_by_field:
        [C, B, B], where entry [c, i, j] is the squared pointwise L2
        distance between reference snapshot i and donor snapshot j.

    marginal_w2_by_field:
        [C, B, B], where entry [c, i, j] is the per-field empirical W2^2
        between reference snapshot i and donor snapshot j.

    global_reference_norm_by_field:
        [C], denominator used by relative_l2_per_field over the whole batch.
    """
    batch_size, _, num_fields = fields.shape
    squared_l2_by_field = torch.empty(
        num_fields,
        batch_size,
        batch_size,
        device=fields.device,
        dtype=fields.dtype,
    )
    marginal_w2_by_field = torch.empty_like(squared_l2_by_field)

    for field_index in range(num_fields):
        values = fields[:, :, field_index]
        norms = values.square().sum(dim=1)
        gram = values @ values.transpose(0, 1)
        squared_distances = (
            norms[:, None] + norms[None, :] - 2.0 * gram
        ).clamp_min(0.0)
        squared_l2_by_field[field_index] = squared_distances

        sorted_values = torch.sort(values, dim=1).values
        sorted_norms = sorted_values.square().sum(dim=1)
        sorted_gram = sorted_values @ sorted_values.transpose(0, 1)
        squared_sorted_distances = (
            sorted_norms[:, None]
            + sorted_norms[None, :]
            - 2.0 * sorted_gram
        ).clamp_min(0.0)
        marginal_w2_by_field[field_index] = (
            squared_sorted_distances / values.shape[1]
        )

    global_reference_norm_by_field = (
        fields.square().sum(dim=(0, 1)).sqrt().clamp_min(EPS)
    )

    return (
        squared_l2_by_field,
        marginal_w2_by_field,
        global_reference_norm_by_field,
    )


def score_snapshot_permutation(
    permutation: torch.Tensor,
    squared_l2_by_field: torch.Tensor,
    marginal_w2_by_field: torch.Tensor,
    global_reference_norm_by_field: torch.Tensor,
) -> Test3Candidate:
    """Score one whole-snapshot permutation without rerunning the GFT."""
    batch_size = permutation.numel()
    reference_indices = torch.arange(
        batch_size,
        device=permutation.device,
    )

    selected_l2_squared = squared_l2_by_field[
        :,
        reference_indices,
        permutation,
    ].sum(dim=1)
    relative_l2_per_field_value = (
        selected_l2_squared.sqrt() / global_reference_norm_by_field
    )
    relative_l2_mean = float(
        relative_l2_per_field_value.mean().detach().cpu()
    )

    selected_self = marginal_w2_by_field[
        :,
        reference_indices,
        permutation,
    ]
    self_marginal_error = float(selected_self.mean().detach().cpu())

    return Test3Candidate(
        permutation=permutation.detach().cpu(),
        relative_l2_mean=relative_l2_mean,
        self_marginal_error=self_marginal_error,
    )


def generate_test3_candidates(
    fields: torch.Tensor,
    num_candidates: int,
    seed: int,
) -> list[Test3Candidate]:
    """Generate and cheaply score candidate whole-snapshot permutations."""
    (
        squared_l2_by_field,
        marginal_w2_by_field,
        global_reference_norm_by_field,
    ) = precompute_snapshot_pair_costs(fields)

    batch_size = fields.shape[0]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    candidates: list[Test3Candidate] = []
    seen: set[tuple[int, ...]] = set()

    # Add every cyclic shift first for deterministic coverage.
    for shift in range(1, batch_size):
        permutation = torch.roll(
            torch.arange(batch_size),
            shifts=shift,
        )
        key = tuple(int(value) for value in permutation.tolist())
        seen.add(key)
        candidates.append(
            score_snapshot_permutation(
                permutation.to(fields.device),
                squared_l2_by_field,
                marginal_w2_by_field,
                global_reference_norm_by_field,
            )
        )

    while len(candidates) < num_candidates:
        permutation = torch.randperm(
            batch_size,
            generator=generator,
        )
        key = tuple(int(value) for value in permutation.tolist())
        if key in seen or torch.equal(
            permutation,
            torch.arange(batch_size),
        ):
            continue
        seen.add(key)
        candidates.append(
            score_snapshot_permutation(
                permutation.to(fields.device),
                squared_l2_by_field,
                marginal_w2_by_field,
                global_reference_norm_by_field,
            )
        )

    return candidates


def relative_gap(value_a: float, value_b: float, eps: float = EPS) -> float:
    denominator = max(0.5 * (abs(value_a) + abs(value_b)), eps)
    return abs(value_a - value_b) / denominator


def select_test3_pair(
    candidates: list[Test3Candidate],
    self_relative_tolerance: float,
    minimum_l2_gap: float,
) -> tuple[Test3Candidate, Test3Candidate, dict[str, float]]:
    """
    Select two permutations with similar self-error and different L2 error.
    """
    best_pair: tuple[Test3Candidate, Test3Candidate] | None = None
    best_l2_gap = -1.0
    best_self_gap = float("inf")

    for index_a, candidate_a in enumerate(candidates):
        for candidate_b in candidates[index_a + 1 :]:
            self_gap = relative_gap(
                candidate_a.self_marginal_error,
                candidate_b.self_marginal_error,
            )
            l2_gap = abs(
                candidate_a.relative_l2_mean
                - candidate_b.relative_l2_mean
            )

            if self_gap <= self_relative_tolerance and l2_gap > best_l2_gap:
                best_pair = (candidate_a, candidate_b)
                best_l2_gap = l2_gap
                best_self_gap = self_gap

    if best_pair is None:
        # Fallback: prioritize self matching, then L2 separation.
        fallback_score = float("inf")
        for index_a, candidate_a in enumerate(candidates):
            for candidate_b in candidates[index_a + 1 :]:
                self_gap = relative_gap(
                    candidate_a.self_marginal_error,
                    candidate_b.self_marginal_error,
                )
                l2_gap = abs(
                    candidate_a.relative_l2_mean
                    - candidate_b.relative_l2_mean
                )
                score = self_gap - 0.1 * l2_gap
                if score < fallback_score:
                    fallback_score = score
                    best_pair = (candidate_a, candidate_b)
                    best_l2_gap = l2_gap
                    best_self_gap = self_gap

    if best_pair is None:
        raise RuntimeError("Could not select a Test 3 corruption pair.")

    lower_l2, higher_l2 = sorted(
        best_pair,
        key=lambda candidate: candidate.relative_l2_mean,
    )

    metadata = {
        "self_relative_gap": best_self_gap,
        "relative_l2_gap": best_l2_gap,
        "requested_self_relative_tolerance": self_relative_tolerance,
        "requested_minimum_l2_gap": minimum_l2_gap,
        "meets_self_tolerance": float(
            best_self_gap <= self_relative_tolerance
        ),
        "meets_minimum_l2_gap": float(best_l2_gap >= minimum_l2_gap),
    }
    return lower_l2, higher_l2, metadata


# ---------------------------------------------------------------------------
# Visualization utilities
# ---------------------------------------------------------------------------


def field_tensor_to_image(field: torch.Tensor) -> np.ndarray:
    """Convert one flattened field [N] into a [Num_y, Num_x] image."""
    values = field.detach().cpu().numpy().reshape(-1)
    expected_nodes = GRID_NUM_X * GRID_NUM_Y
    if values.size != expected_nodes:
        raise ValueError(
            f"Expected {expected_nodes} spatial values, "
            f"received {values.size}."
        )
    return values.reshape(GRID_NUM_X, GRID_NUM_Y).T


def save_field_triptych(
    fields_reference: torch.Tensor,
    fields_corrupted: torch.Tensor,
    field_index: int,
    snapshot_index: int,
    corruption_title: str,
    output_path: Path,
) -> None:
    """Save original, corrupted, and absolute-difference field maps."""
    field_name = FIELD_NAMES[field_index]
    reference_image = field_tensor_to_image(
        fields_reference[snapshot_index, :, field_index]
    )
    corrupted_image = field_tensor_to_image(
        fields_corrupted[snapshot_index, :, field_index]
    )
    difference_image = np.abs(corrupted_image - reference_image)

    shared_min = min(reference_image.min(), corrupted_image.min())
    shared_max = max(reference_image.max(), corrupted_image.max())

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(16, 4.5),
        constrained_layout=True,
    )

    reference_plot = axes[0].imshow(
        reference_image,
        origin="lower",
        aspect="auto",
        vmin=shared_min,
        vmax=shared_max,
    )
    axes[0].set_title(f"Reference {field_name}")

    axes[1].imshow(
        corrupted_image,
        origin="lower",
        aspect="auto",
        vmin=shared_min,
        vmax=shared_max,
    )
    axes[1].set_title(f"Corrupted {field_name}")

    difference_plot = axes[2].imshow(
        difference_image,
        origin="lower",
        aspect="auto",
    )
    axes[2].set_title("Absolute pointwise difference")

    for axis in axes:
        axis.set_xlabel("x index")
        axis.set_ylabel("y index")

    figure.colorbar(
        reference_plot,
        ax=axes[:2],
        label="Normalized field value",
        shrink=0.85,
    )
    figure.colorbar(
        difference_plot,
        ax=axes[2],
        label="Absolute error",
        shrink=0.85,
    )

    figure.suptitle(
        f"{corruption_title} — snapshot {snapshot_index}",
        fontsize=15,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_marginal_distribution_comparison(
    fields_reference: torch.Tensor,
    fields_corrupted: torch.Tensor,
    field_index: int,
    corruption_title: str,
    output_path: Path,
    num_bins: int = 100,
) -> None:
    """Show marginal densities and their absolute binwise difference."""
    field_name = FIELD_NAMES[field_index]

    reference_values = (
        fields_reference[:, :, field_index]
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )
    corrupted_values = (
        fields_corrupted[:, :, field_index]
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    combined_values = np.concatenate([reference_values, corrupted_values])
    bin_edges = np.histogram_bin_edges(combined_values, bins=num_bins)

    reference_density, _ = np.histogram(
        reference_values,
        bins=bin_edges,
        density=True,
    )
    corrupted_density, _ = np.histogram(
        corrupted_values,
        bins=bin_edges,
        density=True,
    )

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    density_difference = np.abs(reference_density - corrupted_density)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13, 4.5),
        constrained_layout=True,
    )

    axes[0].plot(
        bin_centers,
        reference_density,
        linewidth=2.2,
        label=f"Reference {field_name}",
    )
    axes[0].plot(
        bin_centers,
        corrupted_density,
        linestyle="--",
        linewidth=1.8,
        marker="o",
        markevery=8,
        fillstyle="none",
        label=f"Corrupted {field_name}",
    )
    axes[0].set_title(f"{field_name} marginal distributions")
    axes[0].set_xlabel("Normalized field value")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(bin_centers, density_difference, linewidth=2.0)
    axes[1].set_title("Absolute histogram difference")
    axes[1].set_xlabel("Normalized field value")
    axes[1].set_ylabel("Absolute density difference")
    axes[1].grid(alpha=0.25)
    axes[1].text(
        0.02,
        0.96,
        f"Maximum difference: {density_difference.max():.3e}",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
    )

    figure.suptitle(
        f"{corruption_title}: marginal self-consistency diagnostic",
        fontsize=14,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)



def save_png_safely(
    figure: plt.Figure,
    output_path: Path,
    dpi: int = 180,
) -> None:
    """
    Render the complete figure in memory before writing it to disk.

    This avoids partially written PNG files if figure rendering fails and
    deliberately avoids ``bbox_inches="tight"``, which can produce
    extremely large or unreadable images for multi-panel categorical plots.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    buffer = BytesIO()
    try:
        figure.patch.set_facecolor("white")
        figure.canvas.draw()
        figure.savefig(
            buffer,
            format="png",
            dpi=dpi,
            facecolor="white",
            edgecolor="none",
        )
        png_bytes = buffer.getvalue()
    finally:
        plt.close(figure)
        buffer.close()

    png_signature = b"\x89PNG\r\n\x1a\n"
    if not png_bytes.startswith(png_signature):
        raise RuntimeError(f"Figure did not render as a valid PNG: {output_path}")
    if len(png_bytes) < 1024:
        raise RuntimeError(
            f"Rendered PNG is unexpectedly small ({len(png_bytes)} bytes): "
            f"{output_path}"
        )

    temporary_path = output_path.with_name(
        f"{output_path.stem}.temporary.png"
    )
    temporary_path.write_bytes(png_bytes)
    temporary_path.replace(output_path)

def save_metric_panel(
    labeled_metrics: list[tuple[str, dict[str, Any]]],
    title: str,
    output_path: Path,
) -> None:
    """
    Save separate panels for L2, marginal self-error, and cross-frequency
    mutual error. Each panel has its own vertical scale.
    """
    metric_specs = [
        ("relative_l2_mean", "Mean relative L2"),
        ("self_marginal_error", "Marginal self error"),
        ("crossfreq_mutual_error", "Cross-frequency mutual error"),
    ]

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
        constrained_layout=False,
    )

    labels = [label for label, _ in labeled_metrics]
    display_labels = [label.replace(" ", "\n", 1) for label in labels]

    for axis, (metric_key, metric_title) in zip(axes, metric_specs):
        values = np.asarray(
            [
                float(metrics[metric_key])
                for _, metrics in labeled_metrics
            ],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Non-finite values found for {metric_key}: "
                f"{values.tolist()}"
            )

        bars = axis.bar(display_labels, values, width=0.62)
        axis.set_title(metric_title, fontsize=12)
        axis.set_ylabel("Error")
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        axis.margins(x=0.15)

        maximum = float(np.max(values)) if values.size else 0.0
        if maximum > 0.0:
            upper_limit = maximum * 1.30
            text_offset = maximum * 0.035
        else:
            upper_limit = 1.0
            text_offset = 0.02

        axis.set_ylim(0.0, upper_limit)

        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                float(value) + text_offset,
                f"{float(value):.4g}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        if maximum != 0.0 and (maximum < 1.0e-3 or maximum >= 1.0e4):
            axis.ticklabel_format(
                axis="y",
                style="sci",
                scilimits=(0, 0),
            )

    figure.suptitle(title, fontsize=15, y=0.97)
    figure.tight_layout(rect=(0.02, 0.02, 0.98, 0.91))

    save_png_safely(
        figure=figure,
        output_path=output_path,
        dpi=180,
    )

def save_pairwise_crossfreq_bar_chart(
    labeled_metrics: list[tuple[str, dict[str, Any]]],
    title: str,
    output_path: Path,
) -> None:
    """Show which physical field pairs contribute to mutual error."""
    pair_names = list(
        labeled_metrics[0][1]["crossfreq_pair_errors"].keys()
    )
    x_positions = np.arange(len(pair_names))
    width = 0.8 / max(len(labeled_metrics), 1)

    figure, axis = plt.subplots(figsize=(14, 5.5))

    for series_index, (label, metrics) in enumerate(labeled_metrics):
        values = [
            float(metrics["crossfreq_pair_errors"][pair_name])
            for pair_name in pair_names
        ]
        offset = (
            series_index - 0.5 * (len(labeled_metrics) - 1)
        ) * width
        axis.bar(
            x_positions + offset,
            values,
            width=width,
            label=label,
        )

    axis.set_xticks(x_positions)
    axis.set_xticklabels(pair_names, rotation=35, ha="right")
    axis.set_ylabel("Pair contribution to cross-frequency loss")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def extract_pair_coupling_matrix(
    metrics: dict[str, Any],
    key: str,
    field_pair: tuple[int, int],
) -> np.ndarray:
    q_tensor = np.asarray(metrics[key], dtype=np.float64)
    field_i, field_j = field_pair
    return q_tensor[:, :, field_i, field_j]


def save_crossband_heatmaps(
    metrics: dict[str, Any],
    field_pair: tuple[int, int],
    corruption_title: str,
    output_path: Path,
) -> None:
    """Save reference, corrupted, and absolute-difference Q matrices."""
    field_i, field_j = field_pair
    pair_name = f"{FIELD_NAMES[field_i]}-{FIELD_NAMES[field_j]}"
    band_names = list(metrics["band_names"])

    q_reference = extract_pair_coupling_matrix(
        metrics,
        "Q_reference",
        field_pair,
    )
    q_corrupted = extract_pair_coupling_matrix(
        metrics,
        "Q_corrupted",
        field_pair,
    )
    q_difference = np.abs(q_corrupted - q_reference)

    shared_min = min(q_reference.min(), q_corrupted.min())
    shared_max = max(q_reference.max(), q_corrupted.max())

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(14, 4.5),
        constrained_layout=True,
    )

    reference_plot = axes[0].imshow(
        q_reference,
        origin="lower",
        vmin=shared_min,
        vmax=shared_max,
    )
    axes[0].set_title("Reference coupling")

    axes[1].imshow(
        q_corrupted,
        origin="lower",
        vmin=shared_min,
        vmax=shared_max,
    )
    axes[1].set_title("Corrupted coupling")

    difference_plot = axes[2].imshow(q_difference, origin="lower")
    axes[2].set_title("Absolute coupling difference")

    for axis in axes:
        axis.set_xticks(range(len(band_names)))
        axis.set_yticks(range(len(band_names)))
        axis.set_xticklabels(band_names)
        axis.set_yticklabels(band_names)
        axis.set_xlabel("Band of second field")
        axis.set_ylabel("Band of first field")

    figure.colorbar(
        reference_plot,
        ax=axes[:2],
        label="Normalized cross-band coupling Q",
        shrink=0.85,
    )
    figure.colorbar(
        difference_plot,
        ax=axes[2],
        label="Absolute Q difference",
        shrink=0.85,
    )

    figure.suptitle(
        f"{corruption_title}: {pair_name} cross-band coupling",
        fontsize=14,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_test1_field_grid(
    fields_reference: torch.Tensor,
    common_corruption: torch.Tensor,
    single_corruption: torch.Tensor,
    snapshot_index: int,
    field_indices: tuple[int, ...],
    output_path: Path,
) -> None:
    """Compare reference, coherent mix, and single-field mix."""
    figure, axes = plt.subplots(
        len(field_indices),
        3,
        figsize=(14, 4.0 * len(field_indices)),
        constrained_layout=True,
        squeeze=False,
    )

    column_titles = [
        "Reference",
        "All-field donor mix",
        f"{FIELD_NAMES[CO_FIELD_INDEX]}-only donor mix",
    ]
    variants = [fields_reference, common_corruption, single_corruption]

    for row, field_index in enumerate(field_indices):
        images = [
            field_tensor_to_image(
                variant[snapshot_index, :, field_index]
            )
            for variant in variants
        ]
        shared_min = min(image.min() for image in images)
        shared_max = max(image.max() for image in images)

        for column, image in enumerate(images):
            plot = axes[row, column].imshow(
                image,
                origin="lower",
                aspect="auto",
                vmin=shared_min,
                vmax=shared_max,
            )
            axes[row, column].set_title(
                f"{column_titles[column]} — {FIELD_NAMES[field_index]}"
            )
            axes[row, column].set_xlabel("x index")
            axes[row, column].set_ylabel("y index")

        figure.colorbar(
            plot,
            ax=axes[row, :],
            label="Normalized field value",
            shrink=0.78,
        )

    figure.suptitle(
        "Test 1: matched L2, different cross-frequency mutual consistency",
        fontsize=15,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_test3_field_grid(
    fields_reference: torch.Tensor,
    lower_l2_fields: torch.Tensor,
    higher_l2_fields: torch.Tensor,
    snapshot_index: int,
    field_indices: tuple[int, ...],
    output_path: Path,
) -> None:
    """Compare reference with lower- and higher-L2 snapshot permutations."""
    figure, axes = plt.subplots(
        len(field_indices),
        3,
        figsize=(14, 4.0 * len(field_indices)),
        constrained_layout=True,
        squeeze=False,
    )

    column_titles = [
        "Reference snapshot",
        "Lower-L2 donor snapshot",
        "Higher-L2 donor snapshot",
    ]
    variants = [fields_reference, lower_l2_fields, higher_l2_fields]

    for row, field_index in enumerate(field_indices):
        images = [
            field_tensor_to_image(
                variant[snapshot_index, :, field_index]
            )
            for variant in variants
        ]
        shared_min = min(image.min() for image in images)
        shared_max = max(image.max() for image in images)

        for column, image in enumerate(images):
            plot = axes[row, column].imshow(
                image,
                origin="lower",
                aspect="auto",
                vmin=shared_min,
                vmax=shared_max,
            )
            axes[row, column].set_title(
                f"{column_titles[column]} — {FIELD_NAMES[field_index]}"
            )
            axes[row, column].set_xlabel("x index")
            axes[row, column].set_ylabel("y index")

        figure.colorbar(
            plot,
            ax=axes[row, :],
            label="Normalized field value",
            shrink=0.78,
        )

    figure.suptitle(
        "Test 3: different L2, similar self and mutual consistency",
        fontsize=15,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_all_tests_summary(
    test1_metrics: list[tuple[str, dict[str, Any]]],
    test2_metrics: list[tuple[str, dict[str, Any]]],
    test3_metrics: list[tuple[str, dict[str, Any]]],
    output_path: Path,
) -> None:
    """Create one readable summary figure for all three tests."""
    test_groups = [
        (
            "Test 1\nMatched L2,\ndifferent mutual",
            test1_metrics,
        ),
        (
            "Test 2\nMatched self,\ndifferent mutual",
            test2_metrics,
        ),
        (
            "Test 3\nDifferent L2,\nmatched self and mutual",
            test3_metrics,
        ),
    ]

    metric_specs = [
        ("relative_l2_mean", "Relative L2"),
        ("self_marginal_error", "Marginal self error"),
        (
            "crossfreq_mutual_error",
            "Cross-frequency mutual error",
        ),
    ]

    figure, axes = plt.subplots(
        3,
        3,
        figsize=(16, 12),
        constrained_layout=False,
    )

    for row, (test_title, labeled_metrics) in enumerate(test_groups):
        labels = [label for label, _ in labeled_metrics]
        display_labels = [
            label.replace(" ", "\n", 1)
            for label in labels
        ]

        for column, (metric_key, metric_title) in enumerate(metric_specs):
            axis = axes[row, column]
            values = np.asarray(
                [
                    float(metrics[metric_key])
                    for _, metrics in labeled_metrics
                ],
                dtype=np.float64,
            )

            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"Non-finite values in Test {row + 1}, "
                    f"metric {metric_key}: {values.tolist()}"
                )

            bars = axis.bar(display_labels, values, width=0.62)
            axis.set_title(metric_title, fontsize=11)
            axis.grid(axis="y", alpha=0.25)
            axis.set_axisbelow(True)
            axis.margins(x=0.15)

            maximum = float(np.max(values)) if values.size else 0.0
            if maximum > 0.0:
                upper_limit = maximum * 1.32
                text_offset = maximum * 0.04
            else:
                upper_limit = 1.0
                text_offset = 0.02

            axis.set_ylim(0.0, upper_limit)

            for bar, value in zip(bars, values):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    float(value) + text_offset,
                    f"{float(value):.3g}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            if maximum != 0.0 and (
                maximum < 1.0e-3 or maximum >= 1.0e4
            ):
                axis.ticklabel_format(
                    axis="y",
                    style="sci",
                    scilimits=(0, 0),
                )

            if column == 0:
                axis.set_ylabel(f"{test_title}\n\nError", fontsize=10)
            else:
                axis.set_ylabel("Error")

    figure.suptitle(
        (
            "Controlled corruption tests: separating data error, "
            "self-consistency, and mutual consistency"
        ),
        fontsize=16,
        y=0.98,
    )

    figure.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.07,
        top=0.91,
        wspace=0.32,
        hspace=0.55,
    )

    save_png_safely(
        figure=figure,
        output_path=output_path,
        dpi=180,
    )


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def metrics_without_large_tensors(metrics: dict[str, Any]) -> dict[str, Any]:
    """Remove Q tensors from compact console summaries."""
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"Q_corrupted", "Q_reference"}
    }


def save_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Controlled corruption evaluation")
    print("=" * 78)
    print(f"Device: {device}")
    print(f"Dataset: {DATA_PATH}")
    print(f"Dataset statistics: {STATS_PATH}")
    print(f"Graph basis: {GRAPH_BASIS_PATH}")

    _, fields_true, time_indices = load_ground_truth_batch(NUM_SNAPSHOTS)

    require_file(GRAPH_BASIS_PATH, "Graph basis")
    graph_basis, frequency_bands = load_cross_spectral_graph_basis(
        GRAPH_BASIS_PATH
    )

    fields_true = fields_true.to(device)
    graph_basis = graph_basis.to(device)
    frequency_bands = {
        name: indices.to(device)
        for name, indices in frequency_bands.items()
    }

    print(f"Ground-truth shape: {tuple(fields_true.shape)}")
    print(f"Graph basis shape: {tuple(graph_basis.shape)}")
    print(
        "Frequency bands: ",
        {
            name: int(indices.numel())
            for name, indices in frequency_bands.items()
        },
    )
    print(f"Selected time indices: {time_indices.tolist()}")

    # ------------------------------------------------------------------
    # Identity control
    # ------------------------------------------------------------------

    identity_metrics = evaluate_corruption(
        fields_corrupted=fields_true.clone(),
        fields_reference=fields_true,
        graph_basis=graph_basis,
        frequency_bands=frequency_bands,
    )

    print("\n" + "-" * 78)
    print("Identity control")
    print("-" * 78)
    print(json.dumps(metrics_without_large_tensors(identity_metrics), indent=2))

    # ------------------------------------------------------------------
    # TEST 1
    # Similar L2, different cross-frequency mutual consistency
    # ------------------------------------------------------------------

    print("\n" + "=" * 78)
    print("TEST 1: Similar L2, different cross-frequency mutual consistency")
    print("=" * 78)

    donor_permutation = make_cyclic_donor_permutation(
        batch_size=fields_true.shape[0],
        seed=SEED + 101,
        device=device,
    )

    (
        all_field_mix,
        co_only_mix,
        test1_metadata,
    ) = build_matched_l2_corruptions(
        fields=fields_true,
        donor_permutation=donor_permutation,
        field_index=CO_FIELD_INDEX,
        target_fraction=TEST1_MATCHED_L2_FRACTION,
    )

    all_field_mix_metrics = evaluate_corruption(
        fields_corrupted=all_field_mix,
        fields_reference=fields_true,
        graph_basis=graph_basis,
        frequency_bands=frequency_bands,
    )
    co_only_mix_metrics = evaluate_corruption(
        fields_corrupted=co_only_mix,
        fields_reference=fields_true,
        graph_basis=graph_basis,
        frequency_bands=frequency_bands,
    )

    test1_l2_gap = relative_gap(
        float(all_field_mix_metrics["relative_l2_mean"]),
        float(co_only_mix_metrics["relative_l2_mean"]),
    )
    test1_mutual_gap = abs(
        float(all_field_mix_metrics["crossfreq_mutual_error"])
        - float(co_only_mix_metrics["crossfreq_mutual_error"])
    )
    test1_metadata.update(
        {
            "achieved_relative_l2_gap": test1_l2_gap,
            "achieved_crossfreq_mutual_gap": test1_mutual_gap,
            "donor_permutation": donor_permutation.detach().cpu().tolist(),
        }
    )

    print("All-field donor mix:")
    print(
        json.dumps(
            metrics_without_large_tensors(all_field_mix_metrics),
            indent=2,
        )
    )
    print("CO-only donor mix:")
    print(
        json.dumps(
            metrics_without_large_tensors(co_only_mix_metrics),
            indent=2,
        )
    )
    print(f"Matched-L2 relative gap: {test1_l2_gap:.3e}")
    print(f"Cross-frequency mutual-error gap: {test1_mutual_gap:.3e}")

    test1_dir = OUTPUT_DIR / "test_1_matched_l2_different_mutual"
    save_test1_field_grid(
        fields_reference=fields_true,
        common_corruption=all_field_mix,
        single_corruption=co_only_mix,
        snapshot_index=VIS_SNAPSHOT_INDEX,
        field_indices=(CO_FIELD_INDEX, T_FIELD_INDEX),
        output_path=test1_dir / "field_comparison.png",
    )
    save_metric_panel(
        labeled_metrics=[
            ("All-field mix", all_field_mix_metrics),
            ("CO-only mix", co_only_mix_metrics),
        ],
        title="Test 1: matched L2, different cross-frequency mutual consistency",
        output_path=test1_dir / "metric_comparison.png",
    )
    save_pairwise_crossfreq_bar_chart(
        labeled_metrics=[
            ("All-field mix", all_field_mix_metrics),
            ("CO-only mix", co_only_mix_metrics),
        ],
        title="Test 1: pairwise cross-frequency error",
        output_path=test1_dir / "pairwise_crossfreq_errors.png",
    )
    save_crossband_heatmaps(
        metrics=all_field_mix_metrics,
        field_pair=HEATMAP_FIELD_PAIR,
        corruption_title="Test 1A — all-field donor mix",
        output_path=test1_dir / "all_field_mix_co_t_coupling.png",
    )
    save_crossband_heatmaps(
        metrics=co_only_mix_metrics,
        field_pair=HEATMAP_FIELD_PAIR,
        corruption_title="Test 1B — CO-only donor mix",
        output_path=test1_dir / "co_only_mix_co_t_coupling.png",
    )

    # ------------------------------------------------------------------
    # TEST 2
    # Similar marginal self-consistency, different mutual consistency
    # ------------------------------------------------------------------

    print("\n" + "=" * 78)
    print("TEST 2: Similar self-consistency, different mutual consistency")
    print("=" * 78)

    co_spatial_shuffle = spatially_shuffle_single_field(
        fields=fields_true,
        field_index=CO_FIELD_INDEX,
        seed=SEED + 202,
    )
    co_spatial_shuffle_metrics = evaluate_corruption(
        fields_corrupted=co_spatial_shuffle,
        fields_reference=fields_true,
        graph_basis=graph_basis,
        frequency_bands=frequency_bands,
    )

    print("Identity control:")
    print(json.dumps(metrics_without_large_tensors(identity_metrics), indent=2))
    print("CO spatial shuffle:")
    print(
        json.dumps(
            metrics_without_large_tensors(co_spatial_shuffle_metrics),
            indent=2,
        )
    )

    test2_dir = OUTPUT_DIR / "test_2_matched_self_different_mutual"
    save_field_triptych(
        fields_reference=fields_true,
        fields_corrupted=co_spatial_shuffle,
        field_index=CO_FIELD_INDEX,
        snapshot_index=VIS_SNAPSHOT_INDEX,
        corruption_title="Test 2 — CO spatial shuffle",
        output_path=test2_dir / "co_field_comparison.png",
    )
    save_marginal_distribution_comparison(
        fields_reference=fields_true,
        fields_corrupted=co_spatial_shuffle,
        field_index=CO_FIELD_INDEX,
        corruption_title="Test 2 — CO spatial shuffle",
        output_path=test2_dir / "co_marginal_distribution.png",
    )
    save_metric_panel(
        labeled_metrics=[
            ("Identity", identity_metrics),
            ("CO shuffle", co_spatial_shuffle_metrics),
        ],
        title="Test 2: matched marginal self-consistency, different mutual consistency",
        output_path=test2_dir / "metric_comparison.png",
    )
    save_pairwise_crossfreq_bar_chart(
        labeled_metrics=[
            ("Identity", identity_metrics),
            ("CO shuffle", co_spatial_shuffle_metrics),
        ],
        title="Test 2: pairwise cross-frequency error after CO shuffle",
        output_path=test2_dir / "pairwise_crossfreq_errors.png",
    )
    save_crossband_heatmaps(
        metrics=co_spatial_shuffle_metrics,
        field_pair=HEATMAP_FIELD_PAIR,
        corruption_title="Test 2 — CO spatial shuffle",
        output_path=test2_dir / "co_t_crossband_coupling.png",
    )

    # ------------------------------------------------------------------
    # TEST 3
    # Different L2, similar self and mutual consistency
    # ------------------------------------------------------------------

    print("\n" + "=" * 78)
    print("TEST 3: Different L2, similar self and mutual consistency")
    print("=" * 78)

    test3_candidates = generate_test3_candidates(
        fields=fields_true,
        num_candidates=TEST3_NUM_CANDIDATES,
        seed=SEED + 303,
    )
    lower_l2_candidate, higher_l2_candidate, test3_metadata = (
        select_test3_pair(
            candidates=test3_candidates,
            self_relative_tolerance=TEST3_SELF_RELATIVE_TOLERANCE,
            minimum_l2_gap=TEST3_MIN_L2_GAP,
        )
    )

    lower_permutation = lower_l2_candidate.permutation.to(device)
    higher_permutation = higher_l2_candidate.permutation.to(device)
    lower_l2_fields = fields_true[lower_permutation]
    higher_l2_fields = fields_true[higher_permutation]

    lower_l2_metrics = evaluate_corruption(
        fields_corrupted=lower_l2_fields,
        fields_reference=fields_true,
        graph_basis=graph_basis,
        frequency_bands=frequency_bands,
    )
    higher_l2_metrics = evaluate_corruption(
        fields_corrupted=higher_l2_fields,
        fields_reference=fields_true,
        graph_basis=graph_basis,
        frequency_bands=frequency_bands,
    )

    test3_metadata.update(
        {
            "num_candidates": len(test3_candidates),
            "lower_l2_permutation": lower_permutation.detach().cpu().tolist(),
            "higher_l2_permutation": higher_permutation.detach().cpu().tolist(),
            "achieved_self_relative_gap_after_full_evaluation": relative_gap(
                float(lower_l2_metrics["self_marginal_error"]),
                float(higher_l2_metrics["self_marginal_error"]),
            ),
            "achieved_crossfreq_mutual_gap": abs(
                float(lower_l2_metrics["crossfreq_mutual_error"])
                - float(higher_l2_metrics["crossfreq_mutual_error"])
            ),
            "achieved_relative_l2_gap_after_full_evaluation": abs(
                float(lower_l2_metrics["relative_l2_mean"])
                - float(higher_l2_metrics["relative_l2_mean"])
            ),
        }
    )

    print("Lower-L2 whole-snapshot permutation:")
    print(
        json.dumps(
            metrics_without_large_tensors(lower_l2_metrics),
            indent=2,
        )
    )
    print("Higher-L2 whole-snapshot permutation:")
    print(
        json.dumps(
            metrics_without_large_tensors(higher_l2_metrics),
            indent=2,
        )
    )
    print(
        "Test 3 selection metadata:\n"
        + json.dumps(test3_metadata, indent=2)
    )

    test3_dir = OUTPUT_DIR / "test_3_different_l2_matched_self_mutual"
    save_test3_field_grid(
        fields_reference=fields_true,
        lower_l2_fields=lower_l2_fields,
        higher_l2_fields=higher_l2_fields,
        snapshot_index=VIS_SNAPSHOT_INDEX,
        field_indices=(CO_FIELD_INDEX, T_FIELD_INDEX),
        output_path=test3_dir / "field_comparison.png",
    )
    save_metric_panel(
        labeled_metrics=[
            ("Lower L2", lower_l2_metrics),
            ("Higher L2", higher_l2_metrics),
        ],
        title="Test 3: different L2, similar self and mutual consistency",
        output_path=test3_dir / "metric_comparison.png",
    )
    save_pairwise_crossfreq_bar_chart(
        labeled_metrics=[
            ("Lower L2", lower_l2_metrics),
            ("Higher L2", higher_l2_metrics),
        ],
        title="Test 3: pairwise cross-frequency errors remain matched",
        output_path=test3_dir / "pairwise_crossfreq_errors.png",
    )

    # ------------------------------------------------------------------
    # Save one combined summary and all numerical outputs
    # ------------------------------------------------------------------

    save_all_tests_summary(
        test1_metrics=[
            ("All-field mix", all_field_mix_metrics),
            ("CO-only mix", co_only_mix_metrics),
        ],
        test2_metrics=[
            ("Identity", identity_metrics),
            ("CO shuffle", co_spatial_shuffle_metrics),
        ],
        test3_metrics=[
            ("Lower L2", lower_l2_metrics),
            ("Higher L2", higher_l2_metrics),
        ],
        output_path=OUTPUT_DIR / "all_tests_summary.png",
    )

    output_payload = {
        "experiment": "controlled_corruption_suite",
        "num_snapshots": NUM_SNAPSHOTS,
        "time_indices": time_indices.tolist(),
        "field_names": list(FIELD_NAMES),
        "identity_control": identity_metrics,
        "test_1_similar_l2_different_mutual": {
            "description": (
                "All-field donor mixing and CO-only donor mixing are scaled "
                "to have matched mean relative L2."
            ),
            "metadata": test1_metadata,
            "all_field_donor_mix": all_field_mix_metrics,
            "co_only_donor_mix": co_only_mix_metrics,
        },
        "test_2_similar_self_different_mutual": {
            "description": (
                "CO spatial shuffling preserves each snapshot's CO marginal "
                "distribution while disrupting cross-field relationships."
            ),
            "identity": identity_metrics,
            "co_spatial_shuffle": co_spatial_shuffle_metrics,
        },
        "test_3_different_l2_similar_self_mutual": {
            "description": (
                "Two whole-snapshot permutations are selected to closely "
                "match marginal self-error while producing different L2. "
                "Batch-order-invariant cross-frequency statistics remain "
                "matched."
            ),
            "metadata": test3_metadata,
            "lower_l2_permutation": lower_l2_metrics,
            "higher_l2_permutation": higher_l2_metrics,
        },
    }

    output_path = OUTPUT_DIR / "controlled_corruption_results.json"
    save_json(output_payload, output_path)

    print("\n" + "=" * 78)
    print("Controlled corruption suite complete")
    print("=" * 78)
    print(f"Numerical results: {output_path}")
    print(f"Summary figure: {OUTPUT_DIR / 'all_tests_summary.png'}")
    print("Per-test figures:")
    print(f"  Test 1: {test1_dir}")
    print(f"  Test 2: {test2_dir}")
    print(f"  Test 3: {test3_dir}")


if __name__ == "__main__":
    main()