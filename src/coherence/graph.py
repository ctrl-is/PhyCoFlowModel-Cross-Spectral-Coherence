from typing import Optional, Tuple, Dict
import numpy as np
from scipy.spatial import cKDTree
from scipy import sparse
from scipy.sparse.linalg import eigsh

def get_nodes(dataset, use_raw: bool = False) -> np.ndarray:
    """
    Returns graph node coordinates from TurbulentCombustionH5Dataset.

    use_raw = False uses normalized coordinates, matching model input.
    use_raw = True uses physical coordinates, matching plotting/physical geometry.
    """
    coords = dataset.coords_raw if use_raw else dataset.coords
    return coords.cpu().numpy()

def build_knn_edges(coords: np.ndarray, k: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build k Nearest Neighbors edge list from coordinates.

    Returns:
        edge_index: array of shape [E, 2]
        distances: array of shape [E]
    """
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]

    if n < 2:
        raise ValueError("Need at least 2 points to build kNN edges.")

    k = min(k, n - 1)
    tree = cKDTree(coords)

    distances, indices = tree.query(coords, k=k + 1)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    rows = np.repeat(np.arange(n), k)
    cols = indices.reshape(-1)
    dists = distances.reshape(-1)

    edge_index = np.stack([rows, cols], axis=1)

    return edge_index, dists

def build_weighted_matrix(
        coords: np.ndarray, 
        k: int = 16, 
        sigma: Optional[float] = None
        ) -> Tuple[sparse.csr_matrix, float]:
    """
    Builds the weighted adjaceny matrix W from the nodes and edges.

    Returns:
        W: scipy sparse matrix of shape [N, N]
    """
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]

    edge_index, dists = build_knn_edges(coords, k=k)

    if sigma is None:
        sigma = float(np.median(dists[dists > 0]))

    weights = np.exp(-(dists ** 2) / (2 * sigma ** 2 + 1e-12))

    rows = edge_index[:,0]
    cols = edge_index[:,1]

    W = sparse.coo_matrix((weights, (rows, cols)), shape=(n, n))

    W = W.maximum(W.T)

    W.setdiag(0.0)
    W.eliminate_zeros()

    return W.tocsr(), sigma

def build_degree_matrix(W: sparse.spmatrix) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """
    Build sparse degree matrix D from weighted adjacency matrix W.

    Args:
        W: scipy sparse matrix of shape [N, N]

    Returns:
        D: sparse diagonal degree matrix of shape [N, N]
        degrees: dense vector of shape [N]
    """
    degrees = np.asarray(W.sum(axis = 1)).ravel()
    D = sparse.diags(degrees, format = "csr")

    return D, degrees

def get_graph_laplacian(W: sparse.spmatrix) -> sparse.csr_matrix:
    """
    Builds the Graph Laplacian from the degree Matrix D and the weighted adjacency matrix W.

    Args:
        W: scipy sparse matrix of shape [N, N]
    
    Returns:
        L: sparse normalized graph Laplacian of shape [N, N]
    """
    D, degrees = build_degree_matrix(W)
    I = sparse.eye(W.shape[0], format="csr")

    inverse_sqrt = 1.0 / np.sqrt(np.maximum(degrees, 1e-12))
    D_inverse_sqrt = sparse.diags(inverse_sqrt, format="csr")

    L = I - D_inverse_sqrt @ W @ D_inverse_sqrt

    return L.tocsr()

def spectral_decomposition(L: sparse.csr_matrix, num_modes: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the first num_modes eigenvalues/eigenvectors of the graph Laplacian.

    Args:
        L: sparse normalized graph Laplacian of shape [N, N]
        num_modes: number of graph frequencies to keep

    Return:
        eigenvals: [K]
        U: [N, K], columns are eigenvectors
    """
    n = L.shape[0]
    if num_modes <= 0:
        raise ValueError(f"num_modes must be positive, got {num_modes}.")

    k = min(num_modes, n - 2)
    if k <= 0:
        raise ValueError(f"Graph is too small for sparse eigendecomposition. Got n={n}.")

    eigenvals, U = eigsh(L, k=k, which="SM")

    order = np.argsort(eigenvals)
    eigenvals = eigenvals[order]
    U = U[:, order]

    return eigenvals, U

def make_graph_frequency_bands(
        eigenvalues: np.ndarray,
        exclude_zero: bool = True,
        split: str = "thirds"
        ) -> Dict[str, np.ndarray]:
    """
    Split sorted graph frequencies into low/mid/high index bands.

    Args:
        eigenvalues: [K] sorted graph Laplacian eigenvalues.
        exclude_zero: If True, skip the constant/zero mode.
        split: Frequency split rule. Currently supports "thirds".

    Returns:
        Dict mapping band names to frequency-mode indices.
    """
    valid_modes = np.arange(len(eigenvalues))

    if exclude_zero:
        valid_modes = valid_modes[1:]

    n = len(valid_modes)
    first_cut = n // 3
    second_cut =  2 * n // 3

    bands = {
        "low": valid_modes[:first_cut],
        "mid": valid_modes[first_cut:second_cut],
        "high": valid_modes[second_cut:]
    }

    return bands