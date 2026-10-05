import numpy as np

from sklearn.neighbors import NearestNeighbors

# Memory: a library of trusted snapshots (n_library, n_var, n_cell) that can be searched for the ones
# closest to a set of observations.


def init_memory(Q_library, obs_indices, num_neighbors):

    memory_param = {}

    memory_param['Q_library']   = Q_library
    memory_param['obs_indices'] = obs_indices
    memory_param['var_scale']   = np.max(np.abs(Q_library), axis=(0, 2))

    # the library is searched at the observed cells only, with every variable scaled to order one
    library_obs = Q_library[:, :, obs_indices] / memory_param['var_scale'][:, None]

    num_neighbors = min(num_neighbors, Q_library.shape[0])

    memory_param['knn'] = NearestNeighbors(n_neighbors=num_neighbors).fit(library_obs.reshape(Q_library.shape[0], -1))

    return memory_param


def find_neighbors(memory_param, obs):

    obs_scaled = (obs / memory_param['var_scale'][:, None]).reshape(1, -1)

    _, indices = memory_param['knn'].kneighbors(obs_scaled)

    return memory_param['Q_library'][indices[0]]
