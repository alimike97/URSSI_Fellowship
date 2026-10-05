import numpy as np

# Innovation check: mismatch between the snapshot and observations of the true state.
# Unlike the other checks this one needs data from outside the model (sensors, or the FOM in development).


def init_innovation(Q_train, obs_indices):

    innovation_param = {}

    innovation_param['obs_indices'] = obs_indices
    innovation_param['var_scale']   = np.max(np.abs(Q_train), axis=(0, 2))

    return innovation_param


def innovation_check(innovation_param, Q, obs):

    # obs has the shape (n_var, n_obs): the observed state at the cells obs_indices
    mismatch = np.mean(np.abs(Q[:, innovation_param['obs_indices']] - obs), axis=1)

    score = np.max(mismatch / innovation_param['var_scale'])

    return score
