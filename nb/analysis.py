import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import xswap


def auto_to_matrix(edges):
    if isinstance(edges[0][0], str):
        mapped, mapping, _ = xswap.preprocessing.map_str_edges(edges, False)
        return edges_to_matrix(mapped)
    return edges_to_matrix(edges)


def edges_to_matrix(edges, directed=False):
    n_nodes = max(map(max, edges)) + 1
    if not directed:
        edges = list(set(
            list(map(tuple, map(sorted, edges)))
            + list(map(tuple, map(reversed, map(sorted, edges))))
        ))
    sp_mat = scipy.sparse.coo_matrix(
        (np.ones(len(edges)), zip(*edges)), shape=(n_nodes, n_nodes))
    return sp_mat


def normalize(matrix):
    # Normalize adjacency matrix
    rowsums = np.array(matrix.sum(axis=1)).flatten()
    rowsums[rowsums == 0] = 1
    norm_mat = scipy.sparse.diags(1 / rowsums, 0)
    normed = norm_mat @ matrix
    return normed


def normalize_laplacian(matrix):
    '''
    Row normalization for RWR does not give symmetric results. Conversely,
    normalization using a graph Laplacian does.

    Zhou, D., O. Bousquet, T. N. Lal, and J. Weston. "Scholkopf., B.(2004).
    Learning with local and global consistency." In 16th Annual Conf. on NIPS. 2003.
    '''
    diagonal = np.array(matrix.sum(axis=1)).flatten() ** (-1/2)
    if isinstance(matrix, np.ndarray):
        D = np.diag(diagonal)
    elif scipy.sparse.issparse(matrix):
        D = scipy.sparse.diags(diagonal)
    return D@matrix@D


def invertible_rwr(adjacency, restart_prob):
    w = normalize_laplacian(adjacency)
    q = scipy.sparse.identity(w.shape[0]) - (1 - restart_prob) * w
    if scipy.sparse.issparse(q):
        try:
            rwr = restart_prob * scipy.sparse.linalg.inv(q.tocsc())
        except np.linalg.LinAlgError:
            # Use Moore-Penrose pseudoinverse if the matrix is not invertible
            rwr = restart_prob * scipy.linalg.pinv(q.tocsc())
    else:
        try:
            rwr = restart_prob * np.linalg.inv(q)
        except np.linalg.LinAlgError:
            # Use Moore-Penrose pseudoinverse if the matrix is not invertible
            rwr = restart_prob * np.linalg.pinv(q)
    return rwr


def rwr_approx_inv(adjacency, restart_prob, n_iter=20):
    '''
    Approximate matrix inverse using truncated Neumann series.
    R = (1-c)WR + c1 -> R = c(1 - (1-c)W)^(-1)
    Let A = 1 - (1-c)W
    A^(-1) = \sum_{n=0}^\infty (1 - A)^n
    '''
    w = (1-restart_prob) * normalize_laplacian(adjacency)
    q = scipy.sparse.eye(w.shape[0])
    rwr = q
    for i in range(n_iter):
        q = q@w
        rwr += q
    return restart_prob * rwr


def jaccard(adjacency, degree_matrix):
    A2 = adjacency @ adjacency.T
    denom = (degree_matrix - A2.toarray())
    denom[denom == 0] = 1
    return A2 / denom


def compute_directed_degrees(matrix):
    out_degree = np.array(matrix.sum(axis=1), dtype=float).flatten()
    in_degree = np.array(matrix.sum(axis=0), dtype=float).flatten()
    return out_degree, in_degree


def directed_inference(matrix, out_degree, in_degree, n_source):
    '''
    Feature for the link prediction in directed graphs.

    Gasulla, Dario García. "Link prediction in large directed graphs."
    PhD diss., Universitat Politècnica de Catalunya (UPC), 2015.
    '''
    ded = (matrix@matrix)[:n_source].toarray()
    for i, v in enumerate(out_degree[:n_source]):
        if v != 0:
            ded[i] /= v
    ind = (matrix.T@matrix)[:n_source].toarray()
    for i, v in enumerate(in_degree[:n_source]):
        if v != 0:
            ind[i] /= v
    return ded + ind
