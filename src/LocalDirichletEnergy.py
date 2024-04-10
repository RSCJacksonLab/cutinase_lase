'''
Function for calculating local Dirichlet energy
'''

import numpy as np

from numpy.typing import ArrayLike
from scipy import spatial
from sklearn.neighbors import NearestNeighbors


def get_local_dirichlet_energy(rep_arr: ArrayLike, y: ArrayLike):
    '''
    Get the normalised Dirichlet energy for each datapoint given the
    representation array and signals (y). 
    
    Arguments:
    ----------
    rep_arr : ArrayLike
        Array of numerical representations for a protein sequence. 
        Expects (n_seqs, rep_dim)
    y : ArrayLike
        Signals/fitness values of each sequence.

    Returns:
    --------
    local_en_ls : list
        A list of local Dirichlet energies.
    local_edge_ls : list
        A list of connections for each node.
    '''
    # Make kNN
    dist_arr = spatial.distance_matrix(rep_arr, rep_arr)
    k = int(np.sqrt(len(rep_arr)))
    knn_fn = NearestNeighbors(
        n_neighbors=k,
        metric="precomputed"
    ).fit(dist_arr)

    # Determine graph Laplacian
    adj_mat = knn_fn.kneighbors_graph(dist_arr).toarray()

    ## remove self connections
    adj_mat -= np.eye(adj_mat.shape[0])

    ## make adjacency matrix symmetric
    adj_mat = adj_mat + adj_mat.T
    adj_mat[adj_mat > 1] = 1

    # get local dirichlet energy
    local_en_ls = []
    local_edge_ls = []
    for i in range(len(adj_mat)):

        # determine neighbours of the ith node
        neighbour_idx = np.where(adj_mat[i] == 1)
        local_edge_ls.append(neighbour_idx)

        # plus one to account for self
        n = np.sum(adj_mat[i] == 1) + 1

        # make local adj mat
        local_adj_mat = np.zeros((n, n))
        local_adj_mat[:, 0] = 1
        local_adj_mat[0, :] = 1
        local_adj_mat[0, 0] = 0

        # get local signals
        local_y = np.concatenate(
            (y[i].reshape(1, -1), y[neighbour_idx].reshape(1, -1)), 
            axis=1,
        )[0]

        # make diagonal matrix from adjacency
        diag_mat = np.diag(np.sum(local_adj_mat, axis=0))

        # determine graph laplacian
        laplacian = diag_mat - local_adj_mat

        # determine dirichlet energy
        local_dir_en = (local_y @ laplacian) @ local_y.T
        local_dir_en /= len(local_y)

        local_en_ls.append(local_dir_en)
        
    return local_en_ls, local_edge_ls