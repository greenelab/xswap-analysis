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
    rowsums = np.array(matrix.sum(axis=1), dtype=np.float32).flatten()
    rowsums[rowsums == 0] = 1
    norm_mat = scipy.sparse.diags(1 / rowsums, 0, dtype=np.float32)
    normed = norm_mat @ matrix
    return normed


def invertible_rwr(adjacency, restart_prob):
    w = normalize(adjacency)
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
    # 20 iterations is usually sufficient to get maximum absolute difference ~10^-10
    w = (1-restart_prob) * normalize(adjacency)
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
    