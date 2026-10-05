import numpy as np

from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance
from sklearn.neighbors import LocalOutlierFactor

# Out-of-distribution checks: how far is a snapshot from the training snapshots?
# Snapshots are compared in a POD (PCA) latent space fitted on the training window.
# Every check returns a scalar score, where larger means further from the training data.


def snapshot2vector(ood_param, Q):

    # scale each variable to order one and stack them in one vector: (..., n_var * n_cell)
    Q_scaled = Q / ood_param['var_scale'][:, None]

    return Q_scaled.reshape(*Q.shape[:-2], -1)


def init_ood(Q_train, num_modes):

    ood_param = {}

    ood_param['var_scale'] = np.max(np.abs(Q_train), axis=(0, 2))

    X_train = snapshot2vector(ood_param, Q_train)

    num_modes = min(num_modes, X_train.shape[0] - 1)

    ood_param['pca']    = PCA(n_components=num_modes).fit(X_train)
    latent_train        = ood_param['pca'].transform(X_train)

    ood_param['cov']    = EmpiricalCovariance().fit(latent_train)
    ood_param['lof']    = LocalOutlierFactor(n_neighbors=min(20, X_train.shape[0] - 1), novelty=True).fit(latent_train)

    return ood_param


def pod_projection_check(ood_param, Q):

    # relative part of the snapshot that the training basis cannot represent
    pca = ood_param['pca']
    x   = snapshot2vector(ood_param, Q)[None]

    x_centered = x - pca.mean_
    x_residual = x - pca.inverse_transform(pca.transform(x))

    score = np.linalg.norm(x_residual) / np.linalg.norm(x_centered)

    return score


def mahalanobis_check(ood_param, Q):

    # distance from the training cloud in latent space, in units of its standard deviation
    latent = ood_param['pca'].transform(snapshot2vector(ood_param, Q)[None])

    score = np.sqrt(ood_param['cov'].mahalanobis(latent)[0])

    return score


def lof_check(ood_param, Q):

    # local outlier factor in latent space: about 1 inside the training data, larger outside
    latent = ood_param['pca'].transform(snapshot2vector(ood_param, Q)[None])

    score = -ood_param['lof'].score_samples(latent)[0]

    return score
