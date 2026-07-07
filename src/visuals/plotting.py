import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_coherence_pair(
    coherence_pred,
    coherence_target,
    field_i,
    field_j,
    eigenvals=None,
    field_names=None,
    save_path=None,
):
    """
    Plot predicted vs target coherence for one field pair.

    Args:
        coherence_pred:   [K, C, C] torch.Tensor or np.ndarray
        coherence_target: [K, C, C] torch.Tensor or np.ndarray
        field_i: first field index
        field_j: second field index
        eigenvals: optional [K] graph Laplacian eigenvalues
        field_names: optional list of field names
        save_path: optional path to save figure
    """

    if torch.is_tensor(coherence_pred):
        coherence_pred = coherence_pred.cpu().numpy()
    if torch.is_tensor(coherence_target):
        coherence_target = coherence_target.cpu().numpy()
    if eigenvals is not None and torch.is_tensor(eigenvals):
        eigenvals = eigenvals.cpu().numpy()

    K = coherence_pred.shape[0]

    if eigenvals is None:
        x = np.arange(K)
        xlabel = "Graph frequency mode k"
    else:
        x = eigenvals
        xlabel = "Graph Laplacian eigenvalue"

    if field_names is None:
        pair_name = f"Field {field_i} vs Field {field_j}"
    else:
        pair_name = f"{field_names[field_i]} vs {field_names[field_j]}"

    plt.figure(figsize=(8, 5))
    plt.plot(x, coherence_target[:, field_i, field_j], label="Target coherence")
    plt.plot(x, coherence_pred[:, field_i, field_j], label="Predicted coherence", linestyle="--")

    plt.xlabel(xlabel)
    plt.ylabel("Coherence")
    plt.title(f"Graph Cross-Spectral Coherence: {pair_name}")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()

def plot_all_coherence_pairs(
    coherence_pred,
    coherence_target,
    eigenvals=None,
    field_names=None,
    save_dir=None,
):
    """
    Plot predicted vs target coherence for every unique field pair i < j.
    """

    import os

    if torch.is_tensor(coherence_pred):
        coherence_pred = coherence_pred.cpu().numpy()
    if torch.is_tensor(coherence_target):
        coherence_target = coherence_target.cpu().numpy()
    if eigenvals is not None and torch.is_tensor(eigenvals):
        eigenvals = eigenvals.cpu().numpy()

    C = coherence_pred.shape[1]

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for i in range(C):
        for j in range(i + 1, C):
            if save_dir is None:
                save_path = None
            else:
                save_path = os.path.join(save_dir, f"coherence_field_{i}_{j}.png")

            plot_coherence_pair(
                coherence_pred=coherence_pred,
                coherence_target=coherence_target,
                field_i=i,
                field_j=j,
                eigenvals=eigenvals,
                field_names=field_names,
                save_path=save_path,
            )

def plot_coherence_error(
    coherence_pred,
    coherence_target,
    field_i,
    field_j,
    eigenvals=None,
    field_names=None,
    save_path=None,
):
    """
    Plot absolute coherence error across graph frequencies.
    """

    if torch.is_tensor(coherence_pred):
        coherence_pred = coherence_pred.cpu().numpy()
    if torch.is_tensor(coherence_target):
        coherence_target = coherence_target.cpu().numpy()
    if eigenvals is not None and torch.is_tensor(eigenvals):
        eigenvals = eigenvals.cpu().numpy()

    K = coherence_pred.shape[0]

    if eigenvals is None:
        x = np.arange(K)
        xlabel = "Graph frequency mode k"
    else:
        x = eigenvals
        xlabel = "Graph Laplacian eigenvalue"

    error = np.abs(
        coherence_pred[:, field_i, field_j]
        - coherence_target[:, field_i, field_j]
    )

    if field_names is None:
        pair_name = f"Field {field_i} vs Field {field_j}"
    else:
        pair_name = f"{field_names[field_i]} vs {field_names[field_j]}"

    plt.figure(figsize=(8, 5))
    plt.plot(x, error)

    plt.xlabel(xlabel)
    plt.ylabel("Absolute coherence error")
    plt.title(f"Coherence Error: {pair_name}")
    plt.grid(True, alpha=0.3)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()

def plot_band_loss_comparison(model_band_losses, save_path):
    """
    Plot low/mid/high coherence losses for multiple models.

    Args:
        model_band_losses: dict like
            {
                "GL_rbf": {"low": 0.1, "mid": 0.2, "high": 0.3},
                "Perceiver": {"low": 0.08, "mid": 0.25, "high": 0.4},
            }
        save_path: path to save figure.
    """
    bands = ["low", "mid", "high"]
    model_names = list(model_band_losses.keys())

    x = np.arange(len(bands))
    width = 0.8 / max(len(model_names), 1)

    plt.figure(figsize=(8, 5))

    for m, model_name in enumerate(model_names):
        values = [model_band_losses[model_name].get(band, np.nan) for band in bands]
        offset = (m - (len(model_names) - 1) / 2) * width
        plt.bar(x + offset, values, width=width, label=model_name)

    plt.xticks(x, bands)
    plt.xlabel("Graph frequency band")
    plt.ylabel("Coherence loss")
    plt.title("Cross-Spectral Coherence Loss by Frequency Band")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_model_coherence_pair_comparison(
    model_outputs,
    coherence_target,
    field_i,
    field_j,
    eigenvals=None,
    field_names=None,
    save_path=None,
):
    """
    Plot target coherence and multiple model coherence curves for one field pair.

    Args:
        model_outputs: dict mapping model name -> coherence_pred [K, C, C]
        coherence_target: [K, C, C]
        field_i: first field index
        field_j: second field index
        eigenvals: optional [K]
        field_names: optional field names
        save_path: optional output path.
    """
    if torch.is_tensor(coherence_target):
        coherence_target = coherence_target.detach().cpu().numpy()

    if eigenvals is not None and torch.is_tensor(eigenvals):
        eigenvals = eigenvals.detach().cpu().numpy()

    K = coherence_target.shape[0]

    if eigenvals is None:
        x = np.arange(K)
        xlabel = "Graph frequency mode k"
    else:
        x = eigenvals
        xlabel = "Graph Laplacian eigenvalue"

    if field_names is None:
        pair_name = f"Field {field_i} vs Field {field_j}"
    else:
        pair_name = f"{field_names[field_i]} vs {field_names[field_j]}"

    plt.figure(figsize=(8, 5))
    plt.plot(
        x,
        coherence_target[:, field_i, field_j],
        label="Target",
        linewidth=2,
    )

    for model_name, coherence_pred in model_outputs.items():
        if torch.is_tensor(coherence_pred):
            coherence_pred = coherence_pred.detach().cpu().numpy()

        plt.plot(
            x,
            coherence_pred[:, field_i, field_j],
            linestyle="--",
            label=model_name,
        )

    plt.xlabel(xlabel)
    plt.ylabel("Coherence")
    plt.title(f"Model Coherence Comparison: {pair_name}")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()

# Helper Function to handle multiple models
def load_raw_stack(raw_root, max_files=8):
    import os
    import glob
    import numpy as np

    pattern = os.path.join(raw_root, "**", "*coherence_raw.npz")
    raw_paths = sorted(glob.glob(pattern, recursive=True))[-max_files:]

    if len(raw_paths) == 0:
        raise FileNotFoundError(f"No coherence raw files found with pattern: {pattern}")

    fields_pred_list = []
    fields_target_list = []
    coords = None
    field_names = None

    print(f"Using {len(raw_paths)} raw coherence files from {raw_root}:")
    for path in raw_paths:
        print(" -", path)
        data = np.load(path, allow_pickle=True)

        fields_pred_list.append(data["fields_pred"])
        fields_target_list.append(data["fields_target"])

        if coords is None:
            coords = data["coords"]

        if field_names is None and "field_names" in data.files:
            field_names = [str(x) for x in data["field_names"]]

    fields_pred = np.stack(fields_pred_list, axis=0)
    fields_target = np.stack(fields_target_list, axis=0)

    return fields_pred, fields_target, coords, field_names

def compute_pair_losses(coh_pred, coh_target, field_names=None):
    if torch.is_tensor(coh_pred):
        coh_pred = coh_pred.detach().cpu().numpy()
    if torch.is_tensor(coh_target):
        coh_target = coh_target.detach().cpu().numpy()

    C = coh_pred.shape[1]
    pair_losses = {}

    for i in range(C):
        for j in range(i + 1, C):
            name_i = field_names[i] if field_names is not None else f"Field {i}"
            name_j = field_names[j] if field_names is not None else f"Field {j}"
            pair_name = f"{name_i}-{name_j}"

            pair_losses[pair_name] = float(
                np.sum((coh_pred[:, i, j] - coh_target[:, i, j]) ** 2)
            )

    return pair_losses

def plot_model_coherence_summary(model_name, out, field_names, save_path):
    coh_pred = out["coh_pred"]
    coh_target = out["coh_target"]

    if torch.is_tensor(coh_pred):
        coh_pred = coh_pred.detach().cpu().numpy()
    if torch.is_tensor(coh_target):
        coh_target = coh_target.detach().cpu().numpy()

    total_loss = float(np.sum((coh_pred - coh_target) ** 2))

    pair_losses = compute_pair_losses(
        coh_pred,
        coh_target,
        field_names=field_names,
    )

    pair_names = list(pair_losses.keys())
    pair_values = [pair_losses[p] for p in pair_names]

    C = coh_pred.shape[1]

    pred_curves = []
    target_curves = []

    for i in range(C):
        for j in range(i + 1, C):
            pred_curves.append(coh_pred[:, i, j])
            target_curves.append(coh_target[:, i, j])

    pred_curves = np.stack(pred_curves, axis=0)
    target_curves = np.stack(target_curves, axis=0)

    mean_pred = np.mean(pred_curves, axis=0)
    mean_target = np.mean(target_curves, axis=0)
    mean_error = np.mean(np.abs(pred_curves - target_curves), axis=0)

    fig = plt.figure(figsize=(14, 9))

    ax0 = plt.subplot(2, 2, 1)
    ax0.axis("off")
    ax0.text(
        0.5,
        0.55,
        f"{model_name}\n\nTotal coherence loss\n{total_loss:.4e}",
        ha="center",
        va="center",
        fontsize=18,
    )

    ax1 = plt.subplot(2, 2, 2)
    ax1.bar(np.arange(len(pair_names)), pair_values)
    ax1.set_xticks(np.arange(len(pair_names)))
    ax1.set_xticklabels(pair_names, rotation=45, ha="right")
    ax1.set_ylabel("Pair coherence loss")
    ax1.set_title("Field-pair contributions")
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = plt.subplot(2, 2, 3)
    ax2.plot(mean_target, label="Target mean coherence", linewidth=2)
    ax2.plot(mean_pred, label="Predicted mean coherence", linestyle="--")
    ax2.set_xlabel("Graph frequency mode k")
    ax2.set_ylabel("Mean coherence")
    ax2.set_title("Mean coherence across field pairs")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(2, 2, 4)
    ax3.plot(mean_error)
    ax3.set_xlabel("Graph frequency mode k")
    ax3.set_ylabel("Mean absolute coherence error")
    ax3.set_title("Mean coherence error across field pairs")
    ax3.grid(True, alpha=0.3)

    plt.suptitle(f"Cross-Spectral Coherence Summary: {model_name}", fontsize=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def main():
    import os
    import numpy as np
    import torch

    from graph import (
        build_weighted_matrix,
        get_graph_laplacian,
        spectral_decomposition,
        make_graph_frequency_bands,
    )
    from cross_spectral import CrossSpectralConfig, compute_cross_spectral_coherence_loss

    model_roots = {
        "GL_rbf": "PATH_TO_GL_RBF_COHERENCE_RAW_FILES",
        "Perceiver": "PATH_TO_PERCEIVER_COHERENCE_RAW_FILES",
    }

    save_dir = "coherence_plots_eval"
    os.makedirs(save_dir, exist_ok=True)

    num_modes = 64
    k_neighbors = 16

    model_outputs = {}
    shared_coords = None
    shared_field_names = None

    # Load raw predictions for each model.
    for model_name, raw_root in model_roots.items():
        fields_pred_np, fields_target_np, coords, field_names = load_raw_stack(
            raw_root=raw_root,
            max_files=8,
        )

        if shared_coords is None:
            shared_coords = coords
        else:
            if not np.allclose(shared_coords, coords):
                raise ValueError(f"Coords mismatch for model {model_name}.")

        if shared_field_names is None and field_names is not None:
            shared_field_names = field_names

        model_outputs[model_name] = {
            "fields_pred": torch.tensor(fields_pred_np, dtype=torch.float32),
            "fields_target": torch.tensor(fields_target_np, dtype=torch.float32),
        }

    if shared_field_names is None:
        shared_field_names = ["CH4", "CO", "T", "U_1", "p"]

    print("Building graph once from shared coords...")
    W, sigma = build_weighted_matrix(shared_coords, k=k_neighbors)
    print("sigma:", sigma)

    L = get_graph_laplacian(W)
    eigenvals, U_np = spectral_decomposition(L, num_modes=num_modes)
    U = torch.tensor(U_np, dtype=torch.float32)
    bands = make_graph_frequency_bands(eigenvals)

    cfg = CrossSpectralConfig(
        k_neighbors=k_neighbors,
        num_modes=num_modes,
        eps=1e-8,
    )

    coherence_outputs = {}

    # Compute coherence output for each model.
    for model_name, data in model_outputs.items():
        print(f"Computing coherence for {model_name}...")

        out = compute_cross_spectral_coherence_loss(
            fields_pred=data["fields_pred"],
            fields_target=data["fields_target"],
            U=U,
            bands=bands,
            cfg=cfg,
        )

        coherence_outputs[model_name] = out

        model_save_dir = os.path.join(save_dir, model_name)
        os.makedirs(model_save_dir, exist_ok=True)

        plot_all_coherence_pairs(
            coherence_pred=out["coh_pred"],
            coherence_target=out["coh_target"],
            eigenvals=eigenvals,
            field_names=shared_field_names,
            save_dir=model_save_dir,
        )

        C = out["coh_pred"].shape[1]
        for i in range(C):
            for j in range(i + 1, C):
                plot_coherence_error(
                    coherence_pred=out["coh_pred"],
                    coherence_target=out["coh_target"],
                    field_i=i,
                    field_j=j,
                    eigenvals=eigenvals,
                    field_names=shared_field_names,
                    save_path=os.path.join(
                        model_save_dir,
                        f"coherence_error_field_{i}_{j}.png",
                    ),
                )
        plot_model_coherence_summary(
            model_name=model_name,
            out=out,
            field_names=shared_field_names,
            save_path=os.path.join(
                model_save_dir,
                f"{model_name}_coherence_summary.png",
            ),
        )

    print(f"Done. Saved single-model coherence plots to: {save_dir}")

if __name__ == "__main__":
    main()