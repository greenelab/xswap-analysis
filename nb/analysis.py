import numpy as np
import scipy.sparse


def normalize(matrix):
    # Normalize adjacency matrix
    was_sparse = False
    if scipy.sparse.issparse(matrix):
        was_sparse = True
        matrix = matrix.toarray()
    row_sums = (
        matrix
        .sum(axis=1)
        .reshape(matrix.shape[0], 1)
    )
    row_sums[row_sums == 0] = 1
    normalized = np.divide(matrix, row_sums)
    if was_sparse:
        normalized = scipy.sparse.csc_matrix(normalized)
    return normalized

    
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
