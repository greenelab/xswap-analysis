import itertools

import numpy as np
import pandas as pd
import scipy.sparse
import scipy.sparse.linalg
import xswap


def process_edges_to_full_network(edges_df, mapping, allow_loop=False, directed=False):
    '''
    Convert a DataFrame that contains only node pairs where an edge appears in one of 
    the three networks (train, test_recon, test_new) to a DataFrame of all node pairs 
    for nodes appearing in the test network.
    
    Parameters
    ----------
    edges_df : pandas.DataFrame
        Columns should be ['name_a', 'name_b', 'id_a', 'id_b', 'train', 'test_recon', 'test_new'].
        Contains only node pairs with at least one edge, ie. at minimum one of train, test_recon, 
        test_new is 1.
    mapping : dict
        Mapping from name to id
    allow_loop : bool
        Whether to include self-loops as potential edges.
    directed : bool
        Whether edge (a, b) is equivalent to (b, a). If directed, then only source-target node pairs
        will be in the returned DataFrame
        
    Returns
    -------
    pandas.DataFrame
    '''
    reversed_mapping = {v: k for k, v in mapping.items()}
    train_df = edges_df.query('train == 1')
    rel = '<=' if allow_loop else '<'
    
    if directed:
        source_nodes = sorted(set(train_df['id_a']))
        target_nodes = sorted(set(train_df['id_b']))
        id_a, id_b = zip(*itertools.product(source_nodes, target_nodes))
    else:
        nodes = sorted(set(train_df[['id_a', 'id_b']].values.flatten()))
        id_a, id_b = zip(*itertools.product(nodes, nodes))
    
    df = (
        pd.DataFrame()
        .assign(
            id_a=id_a,
            id_b=id_b,
        )
        .query(f'id_a {rel} id_b')
        .merge(edges_df, how='left', on=['id_a', 'id_b'])
        .assign(
            name_a=lambda df: df['id_a'].map(reversed_mapping),
            name_b=lambda df: df['id_b'].map(reversed_mapping),
        )
        .fillna(0)
        .assign(
            train=lambda df: df['train'].astype(int),
            test_recon=lambda df: df['test_recon'].astype(int),
            test_new=lambda df: df['test_new'].astype(int),
        )
        .filter(items=['name_a', 'name_b', 'id_a', 'id_b', 'train', 'test_recon', 'test_new'])
    )
    return df


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
        except:
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
    A2 = adjacency.astype(int) @ adjacency.T.astype(int)
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


def preferential_attachment_index(matrix):
    pref = np.multiply(
        np.repeat(matrix.sum(axis=1), matrix.shape[1], axis=1),
        np.repeat(matrix.sum(axis=0), matrix.shape[0], axis=0)
    )
    return np.array(pref)


def resource_allocation_index(matrix):
    target_degrees = np.repeat(matrix.sum(axis=0), matrix.shape[0], axis=0)
    res = np.multiply(matrix.toarray(), 1 / target_degrees)@matrix.T
    return np.array(res)


def adamic_adar_index(matrix):
    target_degrees = np.repeat(matrix.sum(axis=0), matrix.shape[0], axis=0)
    ad = np.multiply(matrix.toarray(), 1 / np.log(target_degrees + 1))@matrix.T
    return np.array(ad)
