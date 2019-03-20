import numpy as np
import scipy.sparse
import xswap


def auto_to_matrix(edges):
    if isinstance(edges[0][0], str):
        mapped, mapping, _ = xswap.preprocessing.map_str_edges(edges, False)
        return edges_to_matrix(mapped)
    return edges_to_matrix(edges)


def edges_to_matrix(edges):
    n_nodes = max(map(max, edges)) + 1
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


def rwr(normalized_adjacency, start_index, restart_prob, convergence_threshold=1e-6):
    # p(t+1) = (1-r) * W @ p(t) + r * p(0)
    # Setup start position
    p_t = np.zeros((1, normalized_adjacency.shape[0]))
    p_t[0, start_index] = 1
    p_0 = p_t.copy()

    # Iterate RWR until converge
    norm_difference = 1
    while norm_difference > convergence_threshold:
        p_t_1 = (1 - restart_prob) * p_t @ normalized_adjacency + restart_prob * p_0
        norm_difference = np.linalg.norm(p_t_1 - p_t, 1)
        p_t = p_t_1
    return p_t


def all_pairs_rwr(adjacency, restart_prob, convergence_threshold=1e-6):
    normalized_adjacency = normalize(adjacency)

    rwr_matrix = np.zeros(adjacency.shape)

    num_nodes = adjacency.shape[0]
    for seed_index in range(num_nodes):
        rwr_row = rwr(normalized_adjacency, seed_index, restart_prob,
                      convergence_threshold=convergence_threshold)
        rwr_matrix[seed_index, :] = rwr_row
    return rwr_matrix


def full_rwr(adjacency, restart_prob, convergence_threshold=1e-6):
    norm_adj = normalize(adjacency)
    starts = scipy.sparse.identity(norm_adj.shape[0], dtype=float)
    rwr = starts.copy()
    starts *= restart_prob
    norm_difference = 1
    while norm_difference > convergence_threshold:
        rwr_next = (1 - restart_prob) * rwr @ norm_adj + starts
        rwr_diff = rwr_next - rwr
        norm_difference = np.sqrt(rwr_diff.multiply(rwr_diff).sum())
        rwr = rwr_next
    return rwr
