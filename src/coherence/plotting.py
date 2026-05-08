import numpy as np
import matplotlib.pyplot as plt
import torch

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

    plt.show()

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

    plt.show()