import numpy as np

# Ensemble Kalman analysis step (the standard update of the ensemble Kalman filter, for the mean).
# The ensemble gives the background state and its covariance, the observations pull it to the truth.


def enkf_analysis(ensemble, obs, obs_indices, obs_std):

    # ensemble: (n_member, n_var, n_cell)   obs: (n_var, n_obs) at the cells obs_indices
    num_member = ensemble.shape[0]
    shape      = ensemble.shape[1:]

    X      = ensemble.reshape(num_member, -1)
    X_mean = np.mean(X, axis=0)
    A      = X - X_mean

    # the same quantities at the observed cells
    Y      = ensemble[:, :, obs_indices].reshape(num_member, -1)
    Y_mean = np.mean(Y, axis=0)
    S      = Y - Y_mean

    innovation = obs.reshape(-1) - Y_mean

    # solved in ensemble space, where the system is only n_member x n_member
    C       = S @ S.T + (num_member - 1) * obs_std**2 * np.eye(num_member)
    weights = np.linalg.solve(C, S @ innovation)

    X_analysis = X_mean + A.T @ weights

    return X_analysis.reshape(shape)
