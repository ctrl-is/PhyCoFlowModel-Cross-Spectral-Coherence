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

def main():
    import os
    import glob
    import numpy as np
    import torch

    from graph import build_weighted_matrix, get_graph_laplacian, spectral_decomposition
    from cross_spectral import CrossSpectralConfig, combined_spectral_loss

    raw_root = "CHANGE TO BE YOUR ROOT"
    save_dir = "coherence_plots_eval"
    os.makedirs(save_dir, exist_ok=True)

    pattern = os.path.join(raw_root, "**", "*coherence_raw.npz")
    raw_paths = sorted(glob.glob(pattern, recursive=True))

    # Use the latest 8 raw files for now.
    raw_paths = raw_paths[-8:]

    if len(raw_paths) == 0:
        raise FileNotFoundError(f"No coherence raw files found with pattern: {pattern}")

    print(f"Using {len(raw_paths)} raw coherence files:")
    for p in raw_paths:
        print(" -", p)

    fields_pred_list = []
    fields_target_list = []

    coords = None
    field_names = None

    for path in raw_paths:
        data = np.load(path, allow_pickle=True)

        fields_pred_list.append(data["fields_pred"])
        fields_target_list.append(data["fields_target"])

        if coords is None:
            coords = data["coords"]

        if field_names is None and "field_names" in data.files:
            field_names = [str(x) for x in data["field_names"]]

    fields_pred = np.stack(fields_pred_list, axis=0)
    fields_target = np.stack(fields_target_list, axis=0)

    print("fields_pred:", fields_pred.shape)
    print("fields_target:", fields_target.shape)
    print("coords:", coords.shape)
    print("field_names:", field_names)

    fields_pred = torch.tensor(fields_pred, dtype=torch.float32)
    fields_target = torch.tensor(fields_target, dtype=torch.float32)

    num_modes = 64

    print("Building graph...")
    W, sigma = build_weighted_matrix(coords, k=16)
    print("sigma:", sigma)

    print("Building graph Laplacian...")
    L = get_graph_laplacian(W)

    print("Computing graph spectral decomposition...")
    eigenvals, U_np = spectral_decomposition(L, num_modes=num_modes)

    print("U shape:", U_np.shape)
    print("eigenvals shape:", eigenvals.shape)

    U = torch.tensor(U_np, dtype=torch.float32)

    cfg = CrossSpectralConfig(
        k_neighbors=16,
        num_modes=num_modes,
        eps=1e-8,
    )

    print("Computing predicted and target coherence...")
    out = combined_spectral_loss(
        fields_pred=fields_pred,
        fields_target=fields_target,
        U=U,
        cfg=cfg,
    )

    if field_names is None:
        field_names = ["CH4", "CO", "T", "U_1", "p"]

    print("Saving coherence pair plots...")

    plot_all_coherence_pairs(
        coherence_pred=out["coherence_pred"],
        coherence_target=out["coherence_target"],
        eigenvals=eigenvals,
        field_names=field_names,
        save_dir=save_dir,
    )

    print("Saving coherence error plots...")

    C = out["coherence_pred"].shape[1]

    for i in range(C):
        for j in range(i + 1, C):
            plot_coherence_error(
                coherence_pred=out["coherence_pred"],
                coherence_target=out["coherence_target"],
                field_i=i,
                field_j=j,
                eigenvals=eigenvals,
                field_names=field_names,
                save_path=os.path.join(save_dir, f"coherence_error_field_{i}_{j}.png"),
            )

    print(f"Done. Saved coherence plots to: {save_dir}")


if __name__ == "__main__":
    main()