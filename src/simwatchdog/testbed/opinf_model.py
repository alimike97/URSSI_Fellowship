import warnings
import numpy as np
import opinf

from simwatchdog.testbed import pod_func

# Operator inference ROM: a polynomial model for the POD coefficients, learned from data.


def train(config, Q_train, time_train):

    surrogate = config['surrogate']

    surrogate_param = {}

    surrogate_param['pod_param'] = pod_func.init_pod(Q_train, surrogate['num_modes'])

    # opinf works with (n_mode, n_step)
    Q_latent = pod_func.compress(surrogate_param['pod_param'], Q_train).T

    Q_latent, dQdt_latent = opinf.ddt.UniformFiniteDifferencer(time_train, 'ord4').estimate(Q_latent)

    solver = opinf.lstsq.L2Solver(regularizer=surrogate['regularization'])
    model  = opinf.models.ContinuousModel(surrogate['operators'], solver=solver)

    surrogate_param['model'] = model.fit(states=Q_latent, ddts=dQdt_latent)

    return surrogate_param


def rollout(surrogate_param, Q_initial, time):

    pod_param = surrogate_param['pod_param']

    Q_latent_initial = pod_func.compress(pod_param, Q_initial[None])[0]

    # an unstable model stops the time integrator early, which is reported through the NaN below
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Q_latent = surrogate_param['model'].predict(Q_latent_initial, time, method='BDF').T

    Q_surrogate = np.full((time.size, *Q_initial.shape), np.nan)

    Q_surrogate[:Q_latent.shape[0]] = pod_func.decompress(pod_param, Q_latent)

    return Q_surrogate
