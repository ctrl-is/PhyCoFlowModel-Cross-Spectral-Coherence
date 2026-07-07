#!/usr/bin/env python
# Compares Senseiver baseline model and GL_rbf_ENH FFM
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent

# Put both directories at the beginning of sys.path so local repository
# modules are preferred over unrelated installed packages.
# Ensure local cross_spectral modules are searched before src-level modules.
for module_path in (
    str(SRC_DIR),
    str(SCRIPT_DIR),
):
    if module_path in sys.path:
        sys.path.remove(module_path)

    sys.path.insert(
        0,
        module_path,
    )

# ---------------------------------------------------------------------------
# Modules located in src/cross_spectral/
# ---------------------------------------------------------------------------
from eval_coherence import (
    _cross_spectral_coherence_band_metrics,
)

from graph import (
    make_graph_frequency_bands,
)

# ---------------------------------------------------------------------------
# Modules located directly in src/
# ---------------------------------------------------------------------------
from helpers import (
    TurbulentCombustionH5Dataset,
)

from model_finetune import (
    load_pretrained_ffm,
    load_source_config,
)

from model_baseline import (
    build_dataset,
    build_sparse_condition,
    get_baseline_adapter,
    load_yaml,
    safe_torch_load,
    validate_and_normalize_config,
)

# ---------------------------------------------------------------------------
# Plot constants
# ---------------------------------------------------------------------------

EPS = 1e-12

BAND_KEYS = (
    "low",
    "mid",
    "high",
)

BAND_DISPLAY = {
    "low": "large",
    "mid": "medium",
    "high": "small",
}

OFFDIAG_BAND_PAIRS = (
    ("low", "mid", "Low→Mid"),
    ("low", "high", "Low→High"),
    ("mid", "low", "Mid→Low"),
    ("mid", "high", "Mid→High"),
    ("high", "low", "High→Low"),
    ("high", "mid", "High→Mid"),
)

FFM_COLOR = "#0072B2"
SENSEIVER_COLOR = "#D55E00"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Compare FFM and Senseiver graph-band energy ratios.")

    parser.add_argument("--ffm-run-dir", type=str, required=True,
                        help=("FFM run directory or its best.pt/last.pt checkpoint."))
    parser.add_argument("--senseiver-run-dir", type=str, required=True,
                        help=("Senseiver run directory or its best.pt/last.pt checkpoint."))
    parser.add_argument("--graph-basis", type=str, required=True,
                        help="Saved graph basis .pt file.")
    parser.add_argument("--out-dir", type=str, required=True, 
                        help="Directory for comparison plots and numerical outputs.")
    parser.add_argument("--ffm-name", type=str, default="FFM")
    parser.add_argument("--senseiver-name", type=str, default="Senseiver")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--num-snapshots", type=int, default=24,)
    parser.add_argument("--snapshot-indices", type=int, nargs="*", default=None,
                        help=("Explicit positions within the selected split. "
                            "Otherwise, snapshots are distributed across the split."))
    # These defaults match the supplied Senseiver configuration.
    parser.add_argument("--cond-fields", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--n-obs-list", type=int, nargs="+", default=[256, 256])
    # These options apply only to the FFM.
    # Senseiver is a deterministic regressor.
    parser.add_argument("--ffm-n-steps", type=int, default=100)
    parser.add_argument("--ffm-ode-solver", type=str, default="euler",
                        choices=["euler", "heun"])
    parser.add_argument("--use-denorm", action="store_true",
                        help=("Compute graph-band energies in physical units rather "
                            "than normalized model space."))
    parser.add_argument("--device", type=str, default=None,
                        help="For example, cuda:0 or cpu.")
    parser.add_argument("--seed",type=int, default=42)
    parser.add_argument("--denom-tol", type=float, default=1e-12,
                        help=("Numerical epsilon added to ratio denominators."))
    parser.add_argument("--samefreq-ymax", type=float, default=None,
                        help=("Optional fixed y-axis maximum for same-frequency plots."))
    parser.add_argument("--crossfreq-ymax", type=float, default=None,
                        help=("Optional fixed y-axis maximum for cross-frequency plots."))
    parser.add_argument("--dpi", type=int, default=400)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def resolve_run(path_like: str) -> tuple[Path, Path]:
    """
    Resolve either:

        Save_TrainedModel/<run-directory>

    or:

        Save_TrainedModel/<run-directory>/best.pt
    """

    path = Path(path_like).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Model path does not exist: {path}"
        )

    if path.is_file():
        return path.parent, path

    for checkpoint_name in (
        "best.pt",
        "last.pt",
    ):
        checkpoint_path = (
            path
            / checkpoint_name
        )

        if checkpoint_path.exists():
            return path, checkpoint_path

    raise FileNotFoundError(
        f"No best.pt or last.pt was found under {path}"
    )


def load_ffm(run_arg: str, split: str, device: torch.device):
    """
    Load the point-cloud FFM using the same loader already used by the
    original band-energy evaluation.
    """

    run_dir, checkpoint_path = resolve_run(run_arg)

    source_cfg = load_source_config(run_dir)

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    data_path = Path(
        source_cfg.get(
            "data",
            "Dataset/Merged_CH4COTU1P.h5",
        )
    )

    if not data_path.is_absolute():
        data_path = (
            project_root
            / data_path
        )

    if not data_path.exists():
        raise FileNotFoundError(
            f"FFM dataset not found: {data_path}"
        )

    stats_path = (
        run_dir
        / "dataset_stats.pt"
    )

    if not stats_path.exists():
        raise FileNotFoundError(
            f"FFM dataset statistics not found: {stats_path}"
        )

    dataset = TurbulentCombustionH5Dataset(
        h5_path=str(data_path),
        split=split,
        train_ratio=float(
            source_cfg.get(
                "train_ratio",
                0.9,
            )
        ),
        seed=int(
            source_cfg.get(
                "seed",
                42,
            )
        ),
        time_stride=int(
            source_cfg.get(
                "time_stride",
                1,
            )
        ),
        stats_path=str(
            stats_path
        ),
    )

    model, _, _ = load_pretrained_ffm(
        source_run_dir=run_dir,
        checkpoint=checkpoint_path.stem,
        dataset=dataset,
        device=device,
    )

    model.eval()

    # Only FFM hybrid backbones expose these options.
    outer_model = (
        model.module
        if hasattr(
            model,
            "module",
        )
        else model
    )

    backbone = getattr(
        outer_model,
        "model",
        None,
    )

    if (
        backbone is not None
        and hasattr(
            backbone,
            "neighbor_backend",
        )
    ):
        backbone.neighbor_backend = "torch"

    if (
        backbone is not None
        and hasattr(
            backbone,
            "gather_query_chunk_size",
        )
    ):
        backbone.gather_query_chunk_size = 4096

    return (
        model,
        dataset,
        source_cfg,
        run_dir,
        checkpoint_path,
    )


def load_senseiver(run_arg: str, split: str, device: torch.device):
    """
    Load the trained Senseiver using the repository's deterministic
    baseline adapter.
    """

    run_dir, checkpoint_path = resolve_run(run_arg)

    run_cfg_path = (
        run_dir
        / "run_config.yaml"
    )

    if not run_cfg_path.exists():
        raise FileNotFoundError(
            "Senseiver run configuration not found: "
            f"{run_cfg_path}"
        )

    cfg = validate_and_normalize_config(
        load_yaml(
            run_cfg_path
        )
    )

    if cfg["baseline_model"] != "senseiver":
        raise ValueError(
            "Expected a Senseiver run, but "
            f"baseline_model={cfg['baseline_model']!r}."
        )

    if int(cfg["training_stage"]) != 1:
        raise ValueError(
            "Senseiver should use training_stage=1, but "
            f"received training_stage={cfg['training_stage']}."
        )

    stats_path = (
        run_dir
        / "dataset_stats.pt"
    )

    if not stats_path.exists():
        raise FileNotFoundError(
            "Senseiver dataset statistics not found: "
            f"{stats_path}"
        )

    dataset = build_dataset(
        cfg=cfg,
        split=split,
        stats_path=stats_path,
    )

    checkpoint = safe_torch_load(
        checkpoint_path,
        map_location="cpu",
    )

    adapter = get_baseline_adapter(
        "senseiver"
    )

    bundle = adapter.build_for_training(
        cfg=cfg,
        device=device,
        run_dir=run_dir,
        train_set=dataset,
        val_set=dataset,
    )

    # Evaluation only needs model weights. Avoid restoring optimizer and
    # scheduler states, which consume unnecessary GPU/CPU memory.
    checkpoint_for_evaluation = dict(
        checkpoint
    )

    checkpoint_for_evaluation[
        "optimizer"
    ] = None

    checkpoint_for_evaluation[
        "scheduler"
    ] = None

    adapter.load_checkpoint(
        bundle,
        checkpoint_for_evaluation,
    )

    bundle.optimizer = None
    bundle.scheduler = None

    bundle.model.eval()

    print(
        f"Loaded Senseiver checkpoint: {checkpoint_path}"
    )

    return (
        bundle,
        dataset,
        cfg,
        run_dir,
        checkpoint_path,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_datasets(ffm_dataset, senseiver_dataset) -> None:
    """
    The comparison is only valid when both models use the same dataset,
    normalization, split, coordinates, and field ordering.
    """

    problems = []

    if (
        len(ffm_dataset)
        != len(senseiver_dataset)
    ):
        problems.append(
            "split sizes differ: "
            f"FFM={len(ffm_dataset)}, "
            f"Senseiver={len(senseiver_dataset)}"
        )

    if (
        int(ffm_dataset.num_points)
        != int(senseiver_dataset.num_points)
    ):
        problems.append(
            "point counts differ: "
            f"FFM={ffm_dataset.num_points}, "
            f"Senseiver={senseiver_dataset.num_points}"
        )

    if (
        list(ffm_dataset.field_names)
        != list(senseiver_dataset.field_names)
    ):
        problems.append(
            "field names or field ordering differ"
        )

    if (
        hasattr(
            ffm_dataset,
            "indices",
        )
        and hasattr(
            senseiver_dataset,
            "indices",
        )
        and not np.array_equal(
            np.asarray(
                ffm_dataset.indices
            ),
            np.asarray(
                senseiver_dataset.indices
            ),
        )
    ):
        problems.append(
            "train/test split indices differ"
        )

    ffm_coords = (
        ffm_dataset.coords
        .detach()
        .cpu()
    )

    senseiver_coords = (
        senseiver_dataset.coords
        .detach()
        .cpu()
    )

    if (
        ffm_coords.shape
        != senseiver_coords.shape
        or not torch.allclose(
            ffm_coords,
            senseiver_coords,
            rtol=1e-6,
            atol=1e-7,
        )
    ):
        problems.append(
            "normalized coordinates differ"
        )

    ffm_mean = (
        ffm_dataset.mean
        .detach()
        .cpu()
    )

    senseiver_mean = (
        senseiver_dataset.mean
        .detach()
        .cpu()
    )

    if (
        ffm_mean.shape
        != senseiver_mean.shape
        or not torch.allclose(
            ffm_mean,
            senseiver_mean,
            rtol=1e-5,
            atol=1e-7,
        )
    ):
        problems.append(
            "normalization means differ"
        )

    ffm_std = (
        ffm_dataset.std
        .detach()
        .cpu()
    )

    senseiver_std = (
        senseiver_dataset.std
        .detach()
        .cpu()
    )

    if (
        ffm_std.shape
        != senseiver_std.shape
        or not torch.allclose(
            ffm_std,
            senseiver_std,
            rtol=1e-5,
            atol=1e-7,
        )
    ):
        problems.append(
            "normalization standard deviations differ"
        )

    if problems:
        raise ValueError(
            "The FFM and Senseiver evaluations are not directly "
            "comparable:\n  - "
            + "\n  - ".join(
                problems
            )
        )


# ---------------------------------------------------------------------------
# Graph basis
# ---------------------------------------------------------------------------

def load_graph_basis(path: str, num_points: int,device: torch.device):
    graph_path = Path(path).expanduser().resolve()

    if not graph_path.exists():
        raise FileNotFoundError(
            f"Graph basis not found: {graph_path}"
        )

    graph_obj = torch.load(
        graph_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(
        graph_obj,
        dict,
    ):
        raise TypeError(
            "The graph-basis file must contain a dictionary."
        )

    U = graph_obj.get("U")

    if U is None:
        U = graph_obj.get(
            "eigenvectors"
        )

    if U is None:
        U = graph_obj.get(
            "evecs"
        )

    if U is None:
        raise KeyError(
            "Graph basis has no U/eigenvectors/evecs."
        )

    U = torch.as_tensor(
        U,
        dtype=torch.float32,
        device=device,
    )

    if (
        U.ndim != 2
        or int(
            U.shape[0]
        ) != int(
            num_points
        )
    ):
        raise ValueError(
            f"Graph basis shape {tuple(U.shape)} does not "
            f"match N={num_points}."
        )

    saved_bands = graph_obj.get(
        "bands"
    )

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
        eigenvalues = graph_obj.get(
            "eigenvalues"
        )

        if eigenvalues is None:
            eigenvalues = graph_obj.get(
                "evals"
            )

        if eigenvalues is None:
            raise KeyError(
                "Graph basis has neither saved bands nor eigenvalues."
            )

        generated_bands = (
            make_graph_frequency_bands(
                eigenvalues=torch.as_tensor(
                    eigenvalues,
                    dtype=torch.float32,
                    device=device,
                ),
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

    missing = [
        band_name
        for band_name in BAND_KEYS
        if band_name not in bands
    ]

    if missing:
        raise ValueError(
            f"Graph bands are missing {missing}; "
            f"available bands are {list(bands)}."
        )

    return U, bands


# ---------------------------------------------------------------------------
# Shared-sensor reconstruction
# ---------------------------------------------------------------------------

@torch.no_grad()
def same_seed_reconstruct(
    *,
    ffm_model,
    ffm_dataset,
    senseiver_bundle,
    senseiver_dataset,
    snapshot_index: int,
    cond_fields,
    n_obs_list,
    ffm_n_steps: int,
    ffm_ode_solver: str,
    seed: int,
    device: torch.device,
):
    """
    Construct one sparse observation set and give the exact same tensors
    to both the FFM and Senseiver.

    Both models therefore receive identical:
        - sensor locations
        - sensor values
        - observation masks
        - point indices
        - physical-field IDs
    """

    # ------------------------------------------------------------------
    # 1. Load the same snapshot from both datasets
    # ------------------------------------------------------------------

    ffm_sample = ffm_dataset[
        int(snapshot_index)
    ]

    senseiver_sample = senseiver_dataset[
        int(snapshot_index)
    ]

    coords = (
        ffm_sample["coords"]
        .unsqueeze(0)
        .to(device)
    )

    truth = (
        ffm_sample["fields"]
        .unsqueeze(0)
        .to(device)
    )

    senseiver_coords = (
        senseiver_sample["coords"]
        .unsqueeze(0)
        .to(device)
    )

    senseiver_truth = (
        senseiver_sample["fields"]
        .unsqueeze(0)
        .to(device)
    )

    # ------------------------------------------------------------------
    # 2. Ensure the model datasets really contain the same sample
    # ------------------------------------------------------------------

    if not torch.allclose(
        coords,
        senseiver_coords,
        rtol=1e-6,
        atol=1e-7,
    ):
        raise RuntimeError(
            f"Snapshot {snapshot_index}: "
            "FFM and Senseiver coordinates differ."
        )

    if not torch.allclose(
        truth,
        senseiver_truth,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise RuntimeError(
            f"Snapshot {snapshot_index}: "
            "FFM and Senseiver ground truth differs."
        )

    valid_mask = ffm_sample.get(
        "valid_sensor_mask"
    )

    if valid_mask is not None:
        valid_mask = (
            valid_mask
            .unsqueeze(0)
            .to(device)
        )

    # ------------------------------------------------------------------
    # 3. Construct the sparse observations exactly once
    # ------------------------------------------------------------------

    torch.manual_seed(
        int(seed)
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            int(seed)
        )

    (
        obs_coords,
        obs_values,
        obs_mask,
        obs_indices,
        obs_field_ids,
    ) = build_sparse_condition(
        coords_full=coords,
        fields_full=truth,
        cond_fields=cond_fields,
        n_obs_min=n_obs_list,
        n_obs_max=n_obs_list,
        valid_mask=valid_mask,
    )

    actual_observation_count = int(
        obs_mask.sum().item()
    )

    expected_observation_count = int(
        sum(n_obs_list)
    )

    if actual_observation_count != expected_observation_count:
        raise RuntimeError(
            "Unexpected number of observations. "
            f"Expected {expected_observation_count}, "
            f"received {actual_observation_count}."
        )

    # ------------------------------------------------------------------
    # 4. Reconstruct with the FFM
    # ------------------------------------------------------------------

    ffm_sampling_model = (
        ffm_model.module
        if hasattr(
            ffm_model,
            "module",
        )
        else ffm_model
    )

    if not hasattr(
        ffm_sampling_model,
        "sample",
    ):
        raise AttributeError(
            f"Loaded FFM object {type(ffm_sampling_model).__name__} "
            "does not expose sample()."
        )

    ffm_sample_signature = inspect.signature(
        ffm_sampling_model.sample
    )

    ffm_kwargs = {
        "coords": coords,
        "obs_coords": obs_coords,
        "obs_values": obs_values,
        "obs_mask": obs_mask,
        "n_steps": int(ffm_n_steps),
        "clamp_indices": obs_indices,
    }

    if (
        "obs_field_ids"
        in ffm_sample_signature.parameters
    ):
        ffm_kwargs[
            "obs_field_ids"
        ] = obs_field_ids

    elif (
        "cond_field_idx"
        in ffm_sample_signature.parameters
    ):
        valid_field_ids = obs_field_ids[
            obs_mask.bool()
        ]

        unique_field_ids = torch.unique(
            valid_field_ids
        )

        if unique_field_ids.numel() != 1:
            raise ValueError(
                "The loaded FFM checkpoint uses old single-field "
                "conditioning, but multiple conditioned fields were "
                f"requested: {list(cond_fields)}."
            )

        ffm_kwargs[
            "cond_field_idx"
        ] = (
            unique_field_ids
            .view(1)
            .to(device)
        )

    else:
        raise TypeError(
            "The loaded FFM sample() method accepts neither "
            "obs_field_ids nor cond_field_idx."
        )

    if (
        "ode_solver"
        in ffm_sample_signature.parameters
    ):
        ffm_kwargs[
            "ode_solver"
        ] = ffm_ode_solver

    # The FFM starts from a random source field. Give that random draw a
    # reproducible seed separate from the sensor-selection seed.
    torch.manual_seed(
        int(seed) + 1
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            int(seed) + 1
        )

    ffm_recon = ffm_sampling_model.sample(
        **ffm_kwargs
    )

    # ------------------------------------------------------------------
    # 5. Reconstruct with Senseiver using the exact same observations
    # ------------------------------------------------------------------

    senseiver_recon = senseiver_bundle.model(
        query_coords=coords,
        obs_coords=obs_coords,
        obs_values=obs_values,
        obs_mask=obs_mask,
        obs_field_ids=obs_field_ids,
    )

    # ------------------------------------------------------------------
    # 6. Validate output shapes
    # ------------------------------------------------------------------

    if ffm_recon.shape != truth.shape:
        raise RuntimeError(
            "FFM reconstruction shape mismatch: "
            f"recon={tuple(ffm_recon.shape)}, "
            f"truth={tuple(truth.shape)}."
        )

    if senseiver_recon.shape != truth.shape:
        raise RuntimeError(
            "Senseiver reconstruction shape mismatch: "
            f"recon={tuple(senseiver_recon.shape)}, "
            f"truth={tuple(truth.shape)}."
        )

    if not torch.isfinite(
        ffm_recon
    ).all():
        raise RuntimeError(
            f"Snapshot {snapshot_index}: "
            "FFM reconstruction contains NaN or Inf values."
        )

    if not torch.isfinite(
        senseiver_recon
    ).all():
        raise RuntimeError(
            f"Snapshot {snapshot_index}: "
            "Senseiver reconstruction contains NaN or Inf values."
        )

    # ------------------------------------------------------------------
    # 7. Return the common evaluation structure
    # ------------------------------------------------------------------

    shared_payload = {
        "truth": truth,
        "obs_coords": obs_coords,
        "obs_values": obs_values,
        "obs_mask": obs_mask,
        "obs_indices": obs_indices,
        "obs_field_ids": obs_field_ids,
    }

    ffm_output = {
        **shared_payload,
        "recon": ffm_recon,
    }

    senseiver_output = {
        **shared_payload,
        "recon": senseiver_recon,
    }

    return (
        ffm_output,
        senseiver_output,
    )


def denormalize(fields: torch.Tensor, dataset) -> torch.Tensor:
    mean = dataset.mean.to(
        device=fields.device,
        dtype=fields.dtype,
    ).view(
        1,
        1,
        -1,
    )

    std = dataset.std.to(
        device=fields.device,
        dtype=fields.dtype,
    ).view(
        1,
        1,
        -1,
    )

    return (
        fields
        * std
        + mean
    )


# ---------------------------------------------------------------------------
# Ratio utilities
# ---------------------------------------------------------------------------

def ordered_band_indices(band_names):
    normalized = [
        str(name).lower()
        for name in np.asarray(
            band_names
        )
    ]

    missing = [
        band_name
        for band_name in BAND_KEYS
        if band_name not in normalized
    ]

    if missing:
        raise ValueError(
            "Expected low, mid, and high bands. "
            f"Missing={missing}; received={normalized}."
        )

    return {
        band_name: normalized.index(
            band_name
        )
        for band_name in BAND_KEYS
    }


def safe_ratio(numerator, denominator, epsilon: float):
    numerator = np.asarray(
        numerator,
        dtype=np.float64,
    )

    denominator = np.asarray(
        denominator,
        dtype=np.float64,
    )

    if numerator.shape != denominator.shape:
        raise ValueError(
            "Ratio numerator and denominator must have the same shape. "
            f"Got {numerator.shape} and {denominator.shape}."
        )

    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
    )

    result = np.full(
        numerator.shape,
        np.nan,
        dtype=np.float64,
    )

    result[valid] = (
        numerator[valid]
        / (
            denominator[valid]
            + float(epsilon)
        )
    )

    return result

def choose_axis_max(arrays, override):
    if override is not None:
        if override <= 0:
            raise ValueError("A y-axis maximum must be positive.")

        return float(override)

    finite_arrays = []

    for array in arrays:
        values = np.asarray(
            array,
            dtype=np.float64,
        )

        finite = values[
            np.isfinite(values)
        ]

        if finite.size > 0:
            finite_arrays.append(
                finite
            )

    if not finite_arrays:
        return 1.6

    maximum = max(
        float(
            values.max()
        )
        for values in finite_arrays
    )

    return max(
        1.6,
        1.25 * maximum,
    )


def label_bars(
    ax,
    bars,
    values,
    model_name: str,
    y_max: float,
):
    """
    Each bar is explicitly labeled with the model name and ratio value.
    """

    for bar, value in zip(
        bars,
        values,
    ):
        x_position = (
            bar.get_x()
            + bar.get_width()
            / 2
        )

        if np.isfinite(
            value
        ):
            text = (
                f"{model_name}\n"
                f"{value:.2f}"
            )

            y_position = (
                float(
                    value
                )
                + 0.015
                * y_max
            )

        else:
            text = (
                f"{model_name}\n"
                "N/A"
            )

            y_position = (
                0.03
                * y_max
            )

        ax.text(
            x_position,
            y_position,
            text,
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=6,
        )


# ---------------------------------------------------------------------------
# Same-frequency comparison
# ---------------------------------------------------------------------------

def plot_same_frequency_comparison(
    ffm_payload,
    senseiver_payload,
    field_names,
    args,
    output_dir: Path,
):
    ffm_band_indices = ordered_band_indices(
        ffm_payload[
            "band_names"
        ]
    )

    senseiver_band_indices = ordered_band_indices(
        senseiver_payload[
            "band_names"
        ]
    )

    ffm_ratios = safe_ratio(
        ffm_payload[
            "samefreq_energy_pred"
        ],
        ffm_payload[
            "samefreq_energy_true"
        ],
        args.denom_tol,
    )

    senseiver_ratios = safe_ratio(
        senseiver_payload[
            "samefreq_energy_pred"
        ],
        senseiver_payload[
            "samefreq_energy_true"
        ],
        args.denom_tol,
    )

    y_max = choose_axis_max(
        [
            ffm_ratios,
            senseiver_ratios,
        ],
        args.samefreq_ymax,
    )

    x = np.arange(
        len(
            field_names
        )
    )

    width = 0.36

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            14.5,
            5.2,
        ),
        sharey=True,
    )

    for position, band_name in enumerate(
        BAND_KEYS
    ):
        ax = axes[
            position
        ]

        ffm_values = ffm_ratios[
            ffm_band_indices[
                band_name
            ]
        ]

        senseiver_values = senseiver_ratios[
            senseiver_band_indices[
                band_name
            ]
        ]

        ffm_bars = ax.bar(
            x
            - width
            / 2,
            np.nan_to_num(
                ffm_values,
                nan=0.0,
            ),
            width=width,
            label=args.ffm_name,
            color=FFM_COLOR,
            edgecolor="black",
            linewidth=0.5,
        )

        senseiver_bars = ax.bar(
            x
            + width
            / 2,
            np.nan_to_num(
                senseiver_values,
                nan=0.0,
            ),
            width=width,
            label=args.senseiver_name,
            color=SENSEIVER_COLOR,
            edgecolor="black",
            linewidth=0.5,
        )

        for bar, value in zip(
            ffm_bars,
            ffm_values,
        ):
            if not np.isfinite(
                value
            ):
                bar.set_facecolor(
                    "none"
                )

                bar.set_hatch(
                    "//"
                )

        for bar, value in zip(
            senseiver_bars,
            senseiver_values,
        ):
            if not np.isfinite(
                value
            ):
                bar.set_facecolor(
                    "none"
                )

                bar.set_hatch(
                    "\\\\"
                )

        ax.axhline(
            1.0,
            color="black",
            linestyle="--",
            linewidth=1.0,
        )

        ax.set_xticks(
            x
        )

        ax.set_xticklabels(
            field_names,
            rotation=40,
            ha="right",
            fontsize=8,
        )

        ax.set_title(
            f"{BAND_DISPLAY[band_name].capitalize()} scale",
            fontweight="bold",
        )

        ax.set_ylim(
            0,
            y_max,
        )

        ax.grid(
            axis="y",
            alpha=0.3,
        )

        ax.set_axisbelow(
            True
        )

        label_bars(
            ax,
            ffm_bars,
            ffm_values,
            args.ffm_name,
            y_max,
        )

        label_bars(
            ax,
            senseiver_bars,
            senseiver_values,
            args.senseiver_name,
            y_max,
        )

        if position == 0:
            ax.set_ylabel(
                r"$E_b^{\mathrm{model}} / E_b^{\mathrm{GT}}$"
            )

    axes[-1].legend(
        loc="upper right",
        fontsize=8,
    )

    fig.suptitle(
        "Same-Frequency Band-Energy Ratio: FFM vs Senseiver",
        fontweight="bold",
    )

    fig.tight_layout(
        rect=(
            0,
            0,
            1,
            0.95,
        )
    )

    save_path = (
        output_dir
        / "samefreq_band_energy_comparison.png"
    )

    fig.savefig(
        save_path,
        dpi=args.dpi,
        bbox_inches="tight",
    )

    fig.savefig(
        save_path.with_suffix(
            ".pdf"
        ),
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return (
        ffm_ratios,
        senseiver_ratios,
    )


# ---------------------------------------------------------------------------
# Cross-frequency comparison
# ---------------------------------------------------------------------------

def cross_frequency_values(payload, field_pair, tolerance: float):
    band_indices = ordered_band_indices(
        payload[
            "band_names"
        ]
    )

    field_a, field_b = map(
        int,
        field_pair,
    )

    true_values = []
    predicted_values = []
    labels = []

    for (
        source_band,
        target_band,
        display_label,
    ) in OFFDIAG_BAND_PAIRS:

        source_index = band_indices[
            source_band
        ]

        target_index = band_indices[
            target_band
        ]

        true_values.append(
            float(
                np.abs(
                    payload[
                        "S_true"
                    ][
                        source_index,
                        target_index,
                        field_a,
                        field_b,
                    ]
                )
            )
        )

        predicted_values.append(
            float(
                np.abs(
                    payload[
                        "S_pred"
                    ][
                        source_index,
                        target_index,
                        field_a,
                        field_b,
                    ]
                )
            )
        )

        labels.append(
            display_label
        )

    ratios = safe_ratio(
        np.asarray(
            predicted_values,
            dtype=np.float64,
        ),
        np.asarray(
            true_values,
            dtype=np.float64,
        ),
        tolerance,
    )

    return (
        ratios,
        labels,
    )


def draw_cross_frequency_axis(
    ax,
    ffm_payload,
    senseiver_payload,
    field_pair,
    field_names,
    args,
    y_max: float,
    show_legend: bool,
):
    ffm_values, labels = (
        cross_frequency_values(
            ffm_payload,
            field_pair,
            args.denom_tol,
        )
    )

    senseiver_values, senseiver_labels = (
        cross_frequency_values(
            senseiver_payload,
            field_pair,
            args.denom_tol,
        )
    )

    if labels != senseiver_labels:
        raise RuntimeError(
            "Cross-frequency band ordering differs between models."
        )

    x = np.arange(
        len(
            labels
        )
    )

    width = 0.36

    ffm_bars = ax.bar(
        x
        - width
        / 2,
        np.nan_to_num(
            ffm_values,
            nan=0.0,
        ),
        width=width,
        label=args.ffm_name,
        color=FFM_COLOR,
        edgecolor="black",
        linewidth=0.5,
    )

    senseiver_bars = ax.bar(
        x
        + width
        / 2,
        np.nan_to_num(
            senseiver_values,
            nan=0.0,
        ),
        width=width,
        label=args.senseiver_name,
        color=SENSEIVER_COLOR,
        edgecolor="black",
        linewidth=0.5,
    )

    for bar, value in zip(
        ffm_bars,
        ffm_values,
    ):
        if not np.isfinite(
            value
        ):
            bar.set_facecolor(
                "none"
            )

            bar.set_hatch(
                "//"
            )

    for bar, value in zip(
        senseiver_bars,
        senseiver_values,
    ):
        if not np.isfinite(
            value
        ):
            bar.set_facecolor(
                "none"
            )

            bar.set_hatch(
                "\\\\"
            )

    ax.axhline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        labels,
        rotation=35,
        ha="right",
        fontsize=8,
    )

    ax.set_ylim(
        0,
        y_max,
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    ax.set_axisbelow(
        True
    )

    ax.set_ylabel(
        r"$|S^{\mathrm{model}}| / |S^{\mathrm{GT}}|$"
    )

    field_a, field_b = field_pair

    ax.set_title(
        f"{field_names[field_a]}–{field_names[field_b]}",
        fontweight="bold",
    )

    label_bars(
        ax,
        ffm_bars,
        ffm_values,
        args.ffm_name,
        y_max,
    )

    label_bars(
        ax,
        senseiver_bars,
        senseiver_values,
        args.senseiver_name,
        y_max,
    )

    if show_legend:
        ax.legend(
            loc="upper right",
            fontsize=8,
        )

    return (
        ffm_values,
        senseiver_values,
    )


def plot_cross_frequency_comparisons(
    ffm_payload,
    senseiver_payload,
    field_names,
    args,
    output_dir: Path,
):
    field_pairs = [
        tuple(
            map(
                int,
                pair,
            )
        )
        for pair in np.asarray(
            ffm_payload[
                "field_pairs"
            ]
        )
    ]

    all_ffm_values = []

    all_senseiver_values = []

    for field_pair in field_pairs:
        ffm_values, _ = (
            cross_frequency_values(
                ffm_payload,
                field_pair,
                args.denom_tol,
            )
        )

        senseiver_values, _ = (
            cross_frequency_values(
                senseiver_payload,
                field_pair,
                args.denom_tol,
            )
        )

        all_ffm_values.append(
            ffm_values
        )

        all_senseiver_values.append(
            senseiver_values
        )

    all_ffm_values = np.stack(
        all_ffm_values,
        axis=0,
    )

    all_senseiver_values = np.stack(
        all_senseiver_values,
        axis=0,
    )

    y_max = choose_axis_max(
        [
            all_ffm_values,
            all_senseiver_values,
        ],
        args.crossfreq_ymax,
    )

    numerical_results = {}

    # -----------------------------------------------------------------------
    # One comparison plot per physical-field pair
    # -----------------------------------------------------------------------

    for field_pair in field_pairs:
        fig, ax = plt.subplots(
            figsize=(
                10.5,
                4.8,
            )
        )

        (
            ffm_values,
            senseiver_values,
        ) = draw_cross_frequency_axis(
            ax=ax,
            ffm_payload=ffm_payload,
            senseiver_payload=senseiver_payload,
            field_pair=field_pair,
            field_names=field_names,
            args=args,
            y_max=y_max,
            show_legend=True,
        )

        fig.suptitle(
            "Cross-Frequency Band-Energy Ratio: FFM vs Senseiver",
            fontweight="bold",
        )

        fig.tight_layout(
            rect=(
                0,
                0,
                1,
                0.94,
            )
        )

        field_a, field_b = field_pair

        pair_label = (
            f"{field_names[field_a]}_"
            f"{field_names[field_b]}"
        )

        save_path = (
            output_dir
            / (
                "crossfreq_band_energy_comparison_"
                f"{pair_label}.png"
            )
        )

        fig.savefig(
            save_path,
            dpi=args.dpi,
            bbox_inches="tight",
        )

        fig.savefig(
            save_path.with_suffix(
                ".pdf"
            ),
            bbox_inches="tight",
        )

        plt.close(
            fig
        )

        numerical_results[
            pair_label
        ] = {
            "ffm": ffm_values,
            "senseiver": senseiver_values,
        }

    # -----------------------------------------------------------------------
    # Every physical-field pair in one stacked figure
    # -----------------------------------------------------------------------

    fig, axes = plt.subplots(
        len(
            field_pairs
        ),
        1,
        figsize=(
            11,
            4.3
            * len(
                field_pairs
            ),
        ),
        squeeze=False,
        sharey=True,
    )

    for row, field_pair in enumerate(
        field_pairs
    ):
        draw_cross_frequency_axis(
            ax=axes[
                row,
                0,
            ],
            ffm_payload=ffm_payload,
            senseiver_payload=senseiver_payload,
            field_pair=field_pair,
            field_names=field_names,
            args=args,
            y_max=y_max,
            show_legend=(
                row == 0
            ),
        )

    fig.suptitle(
        "Cross-Frequency Band-Energy Ratios for All Field Pairs",
        fontweight="bold",
        y=0.999,
    )

    fig.tight_layout(
        rect=(
            0,
            0,
            1,
            0.995,
        )
    )

    save_path = (
        output_dir
        / "crossfreq_band_energy_comparison_all_pairs.png"
    )

    fig.savefig(
        save_path,
        dpi=args.dpi,
        bbox_inches="tight",
    )

    fig.savefig(
        save_path.with_suffix(
            ".pdf"
        ),
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return numerical_results


# ---------------------------------------------------------------------------
# JSON conversion
# ---------------------------------------------------------------------------

def json_ready(
    value,
):
    if isinstance(
        value,
        np.ndarray,
    ):
        return json_ready(
            value.tolist()
        )

    if isinstance(
        value,
        np.generic,
    ):
        return json_ready(
            value.item()
        )

    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): json_ready(
                item
            )
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            json_ready(
                item
            )
            for item in value
        ]

    if (
        isinstance(
            value,
            float,
        )
        and not np.isfinite(
            value
        )
    ):
        return None

    return value


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    device = torch.device(
        args.device
        or (
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    cond_fields = [
        int(
            field
        )
        for field in args.cond_fields
    ]

    n_obs_list = [
        int(
            count
        )
        for count in args.n_obs_list
    ]

    if (
        len(
            n_obs_list
        ) == 1
        and len(
            cond_fields
        ) > 1
    ):
        n_obs_list = (
            n_obs_list
            * len(
                cond_fields
            )
        )

    if (
        len(
            cond_fields
        )
        != len(
            n_obs_list
        )
    ):
        raise ValueError(
            "--n-obs-list must have length 1 or match --cond-fields."
        )

    torch.manual_seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            args.seed
        )

    print(
        f"Device: {device}"
    )

    print(
        f"Conditioned fields: {cond_fields}"
    )

    print(
        f"Observation counts: {n_obs_list}"
    )

    # -----------------------------------------------------------------------
    # Load both trained models
    # -----------------------------------------------------------------------

    (
        ffm_model,
        ffm_dataset,
        ffm_cfg,
        ffm_run_dir,
        ffm_checkpoint,
    ) = load_ffm(
        run_arg=args.ffm_run_dir,
        split=args.split,
        device=device,
    )

    (
        senseiver_bundle,
        senseiver_dataset,
        senseiver_cfg,
        senseiver_run_dir,
        senseiver_checkpoint,
    ) = load_senseiver(
        run_arg=args.senseiver_run_dir,
        split=args.split,
        device=device,
    )


    # -----------------------------------------------------------------------
    # Check whether evaluation conditioning matches training conditioning
    # -----------------------------------------------------------------------

    # Prefer the actual FFM training-conditioning keys.
    ffm_training_cond_fields = ffm_cfg.get(
        "cond_fields"
    )
    ffm_conditioning_source = "cond_fields"

    # Older FFM configurations may store one conditioned field.
    if ffm_training_cond_fields is None:
        ffm_single_field = ffm_cfg.get(
            "cond_field"
        )

        if ffm_single_field is not None:
            ffm_training_cond_fields = [
                int(ffm_single_field)
            ]
            ffm_conditioning_source = "cond_field"

    # Only use visualization conditioning as a fallback.
    if ffm_training_cond_fields is None:
        ffm_training_cond_fields = ffm_cfg.get(
            "vis_cond_fields"
        )

        if ffm_training_cond_fields is not None:
            ffm_conditioning_source = "vis_cond_fields fallback"
        else:
            ffm_conditioning_source = "not found"

    # Normalize the FFM configuration to list[int].
    if isinstance(
        ffm_training_cond_fields,
        (int, np.integer),
    ):
        ffm_training_cond_fields = [
            int(ffm_training_cond_fields)
        ]

    elif ffm_training_cond_fields is not None:
        ffm_training_cond_fields = [
            int(value)
            for value in ffm_training_cond_fields
        ]

    # Senseiver stores its training conditioning in the nested unified config.
    senseiver_training_cond_fields = [
        int(value)
        for value in senseiver_cfg[
            "shared"
        ][
            "conditioning"
        ][
            "cond_fields"
        ]
    ]

    print(
        "FFM configured conditioned fields "
        f"({ffm_conditioning_source}): "
        f"{ffm_training_cond_fields}"
    )

    print(
        "Senseiver training conditioned fields: "
        f"{senseiver_training_cond_fields}"
    )

    if (
        ffm_training_cond_fields is not None
        and cond_fields != ffm_training_cond_fields
    ):
        print(
            "WARNING: Requested evaluation conditioning "
            f"{cond_fields} differs from the FFM configured conditioning "
            f"{ffm_training_cond_fields}."
        )

    if cond_fields != senseiver_training_cond_fields:
        print(
            "WARNING: Requested evaluation conditioning "
            f"{cond_fields} differs from the Senseiver training conditioning "
            f"{senseiver_training_cond_fields}."
        )


    # -----------------------------------------------------------------------
    # Confirm the datasets are directly comparable
    # -----------------------------------------------------------------------

    validate_datasets(
        ffm_dataset,
        senseiver_dataset,
    )

    number_of_fields = int(
        ffm_dataset.num_fields
    )

    for field_index in cond_fields:
        if not (
            0
            <= field_index
            < number_of_fields
        ):
            raise ValueError(
                f"Conditioned field {field_index} is outside "
                f"[0, {number_of_fields})."
            )

    # -----------------------------------------------------------------------
    # Select shared evaluation snapshots
    # -----------------------------------------------------------------------

    if args.snapshot_indices:
        snapshot_positions = sorted(
            set(
                int(
                    index
                )
                for index in args.snapshot_indices
            )
        )

    else:
        requested_count = min(
            int(
                args.num_snapshots
            ),
            len(
                ffm_dataset
            ),
        )

        snapshot_positions = sorted(
            set(
                np.linspace(
                    0,
                    len(
                        ffm_dataset
                    )
                    - 1,
                    requested_count,
                )
                .astype(
                    int
                )
                .tolist()
            )
        )

    if (
        len(
            snapshot_positions
        )
        < 2
    ):
        raise ValueError(
            "Cross-frequency covariance requires at least two snapshots."
        )

    if (
        min(
            snapshot_positions
        ) < 0
        or max(
            snapshot_positions
        ) >= len(
            ffm_dataset
        )
    ):
        raise IndexError(
            "A requested snapshot position is outside the selected split."
        )

    print(
        f"Snapshot positions: {snapshot_positions}"
    )

    # -----------------------------------------------------------------------
    # Load the shared graph Fourier basis
    # -----------------------------------------------------------------------

    U, bands = load_graph_basis(
        path=args.graph_basis,
        num_points=ffm_dataset.num_points,
        device=device,
    )

    print(
        f"Graph basis shape: {tuple(U.shape)}"
    )

    print(
        "Band sizes:",
        {
            name: int(
                indices.numel()
            )
            for name, indices
            in bands.items()
        },
    )

    # -----------------------------------------------------------------------
    # Reconstruct both models on the exact same observations
    # -----------------------------------------------------------------------

    fields_true_list = []

    fields_ffm_list = []

    fields_senseiver_list = []

    sensor_index_record = {}

    for position, snapshot_index in enumerate(
        snapshot_positions,
        start=1,
    ):
        print(
            f"Reconstructing shared snapshot "
            f"{position}/{len(snapshot_positions)} "
            f"(split position {snapshot_index})"
        )

        reconstruction_seed = (
            args.seed
            + 100003
            * int(
                snapshot_index
            )
        )

        (
            ffm_output,
            senseiver_output,
        ) = same_seed_reconstruct(
            ffm_model=ffm_model,
            ffm_dataset=ffm_dataset,
            senseiver_bundle=senseiver_bundle,
            senseiver_dataset=senseiver_dataset,
            snapshot_index=snapshot_index,
            cond_fields=cond_fields,
            n_obs_list=n_obs_list,
            ffm_n_steps=args.ffm_n_steps,
            ffm_ode_solver=args.ffm_ode_solver,
            seed=reconstruction_seed,
            device=device,
        )

        fields_true = ffm_output[
            "truth"
        ]

        fields_ffm = ffm_output[
            "recon"
        ]

        fields_senseiver = senseiver_output[
            "recon"
        ]

        if args.use_denorm:
            fields_true = denormalize(
                fields_true,
                ffm_dataset,
            )

            fields_ffm = denormalize(
                fields_ffm,
                ffm_dataset,
            )

            fields_senseiver = denormalize(
                fields_senseiver,
                senseiver_dataset,
            )

        fields_true_list.append(
            fields_true
            .detach()
            .cpu()
        )

        fields_ffm_list.append(
            fields_ffm
            .detach()
            .cpu()
        )

        fields_senseiver_list.append(
            fields_senseiver
            .detach()
            .cpu()
        )

        sensor_index_record[
            str(
                snapshot_index
            )
        ] = (
            ffm_output[
                "obs_indices"
            ]
            .detach()
            .cpu()
            .tolist()
        )

    fields_true = torch.cat(
        fields_true_list,
        dim=0,
    ).to(
        device
    )

    fields_ffm = torch.cat(
        fields_ffm_list,
        dim=0,
    ).to(
        device
    )

    fields_senseiver = torch.cat(
        fields_senseiver_list,
        dim=0,
    ).to(
        device
    )

    print(
        f"fields_true shape: {tuple(fields_true.shape)}"
    )

    print(
        f"fields_ffm shape: {tuple(fields_ffm.shape)}"
    )

    print(
        f"fields_senseiver shape: {tuple(fields_senseiver.shape)}"
    )

    # -----------------------------------------------------------------------
    # Compute same-frequency and cross-frequency band energies
    # -----------------------------------------------------------------------

    with torch.no_grad():
        _, ffm_payload = (
            _cross_spectral_coherence_band_metrics(
                fields_true=fields_true,
                fields_pred=fields_ffm,
                U=U,
                bands=bands,
                field_pairs=None,
                eps=EPS,
            )
        )

        _, senseiver_payload = (
            _cross_spectral_coherence_band_metrics(
                fields_true=fields_true,
                fields_pred=fields_senseiver,
                U=U,
                bands=bands,
                field_pairs=None,
                eps=EPS,
            )
        )

    if not np.allclose(
        ffm_payload[
            "samefreq_energy_true"
        ],
        senseiver_payload[
            "samefreq_energy_true"
        ],
        rtol=1e-6,
        atol=1e-8,
    ):
        raise RuntimeError(
            "Same-frequency GT energies differ between the two payloads."
        )

    if not np.allclose(
        ffm_payload[
            "S_true"
        ],
        senseiver_payload[
            "S_true"
        ],
        rtol=1e-6,
        atol=1e-8,
    ):
        raise RuntimeError(
            "Cross-frequency GT covariances differ between the two payloads."
        )

    # -----------------------------------------------------------------------
    # Save comparison plots
    # -----------------------------------------------------------------------

    output_dir = Path(
        args.out_dir
    ).expanduser().resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    field_names = list(
        ffm_dataset.field_names
    )

    (
        samefreq_ratio_ffm,
        samefreq_ratio_senseiver,
    ) = plot_same_frequency_comparison(
        ffm_payload=ffm_payload,
        senseiver_payload=senseiver_payload,
        field_names=field_names,
        args=args,
        output_dir=output_dir,
    )

    cross_frequency_results = (
        plot_cross_frequency_comparisons(
            ffm_payload=ffm_payload,
            senseiver_payload=senseiver_payload,
            field_names=field_names,
            args=args,
            output_dir=output_dir,
        )
    )

    # -----------------------------------------------------------------------
    # Save numerical payload
    # -----------------------------------------------------------------------

    np.savez_compressed(
        (
            output_dir
            / "band_energy_comparison_payload.npz"
        ),

        band_names=ffm_payload[
            "band_names"
        ],

        field_pairs=ffm_payload[
            "field_pairs"
        ],

        samefreq_energy_true=ffm_payload[
            "samefreq_energy_true"
        ],

        samefreq_energy_ffm=ffm_payload[
            "samefreq_energy_pred"
        ],

        samefreq_energy_senseiver=senseiver_payload[
            "samefreq_energy_pred"
        ],

        samefreq_ratio_ffm=(
            samefreq_ratio_ffm
        ),

        samefreq_ratio_senseiver=(
            samefreq_ratio_senseiver
        ),

        crossfreq_energy_true=ffm_payload[
            "crossfreq_energy_true"
        ],

        crossfreq_energy_ffm=ffm_payload[
            "crossfreq_energy_pred"
        ],

        crossfreq_energy_senseiver=senseiver_payload[
            "crossfreq_energy_pred"
        ],

        crossfreq_ratio_ffm=ffm_payload[
            "crossfreq_energy_ratio"
        ],

        crossfreq_ratio_senseiver=senseiver_payload[
            "crossfreq_energy_ratio"
        ],

        S_true=ffm_payload[
            "S_true"
        ],

        S_ffm=ffm_payload[
            "S_pred"
        ],

        S_senseiver=senseiver_payload[
            "S_pred"
        ],
    )

    # -----------------------------------------------------------------------
    # Save metadata
    # -----------------------------------------------------------------------

    metadata = {
        "ffm_name": args.ffm_name,

        "senseiver_name": (
            args.senseiver_name
        ),

        "ffm_run_dir": (
            ffm_run_dir
        ),

        "ffm_checkpoint": (
            ffm_checkpoint
        ),

        "senseiver_run_dir": (
            senseiver_run_dir
        ),

        "senseiver_checkpoint": (
            senseiver_checkpoint
        ),

        "graph_basis": Path(
            args.graph_basis
        ).expanduser().resolve(),

        "split": args.split,

        "snapshot_positions": (
            snapshot_positions
        ),

        "cond_fields": (
            cond_fields
        ),

        "n_obs_list": (
            n_obs_list
        ),

        "sensor_indices": (
            sensor_index_record
        ),

        "ffm_n_steps": (
            args.ffm_n_steps
        ),

        "ffm_ode_solver": (
            args.ffm_ode_solver
        ),

        "use_denorm": (
            args.use_denorm
        ),

        "field_names": (
            field_names
        ),

        "denom_tol": (
            args.denom_tol
        ),

        "cross_frequency_ratios": (
            cross_frequency_results
        ),
    }

    metadata_path = (
        output_dir
        / "band_energy_comparison_metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            json_ready(
                metadata
            ),
            handle,
            indent=2,
            allow_nan=False,
        )

    print(
        "\nBand-energy comparison complete."
    )

    print(
        f"Outputs saved to:\n{output_dir}"
    )

    print(
        "\nSame-frequency comparison:\n"
        f"{output_dir / 'samefreq_band_energy_comparison.png'}"
    )

    print(
        "\nAll-pairs cross-frequency comparison:\n"
        f"{output_dir / 'crossfreq_band_energy_comparison_all_pairs.png'}"
    )


if __name__ == "__main__":
    main()