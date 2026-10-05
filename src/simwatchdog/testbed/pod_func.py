import numpy as np

from sklearn.decomposition import PCA

# POD compression shared by the surrogate models.
# Trajectories have the shape (n_step, n_var, n_cell), latent trajectories (n_step, n_mode).


def init_pod(Q_train, num_modes):

    pod_param = {}

    # scale each variable to order one before stacking them in one vector
    pod_param['var_scale'] = np.max(np.abs(Q_train), axis=(0, 2))
    pod_param['shape']     = Q_train.shape[1:]

    X_train = (Q_train / pod_param['var_scale'][:, None]).reshape(Q_train.shape[0], -1)

    pod_param['pca'] = PCA(n_components=min(num_modes, X_train.shape[0] - 1)).fit(X_train)

    return pod_param


def compress(pod_param, Q):

    X = (Q / pod_param['var_scale'][:, None]).reshape(Q.shape[0], -1)

    return pod_param['pca'].transform(X)


def decompress(pod_param, Q_latent):

    X = pod_param['pca'].inverse_transform(Q_latent)

    return X.reshape(X.shape[0], *pod_param['shape']) * pod_param['var_scale'][:, None]
