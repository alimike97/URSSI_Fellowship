from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.kernel_ridge import KernelRidge

from simwatchdog.testbed import pod_func

# Classical machine learning surrogate: the POD coefficients are regressed against time.


def train(config, Q_train, time_train):

    surrogate = config['surrogate']

    surrogate_param = {}

    surrogate_param['pod_param'] = pod_func.init_pod(Q_train, surrogate['num_modes'])

    Q_latent = pod_func.compress(surrogate_param['pod_param'], Q_train)

    # time is scaled to [0, 1] over the training window
    surrogate_param['time_scale'] = time_train[-1] - time_train[0]
    surrogate_param['time_start'] = time_train[0]

    time_scaled = ((time_train - time_train[0]) / surrogate_param['time_scale'])[:, None]

    if surrogate['regressor'] == 'gpr':

        kernel    = RBF(length_scale=0.1) + WhiteKernel(noise_level=surrogate['regularization'])
        regressor = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=surrogate['seed'])

    if surrogate['regressor'] == 'kernel_ridge':

        regressor = KernelRidge(kernel='rbf', gamma=10.0, alpha=surrogate['regularization'])

    surrogate_param['regressor'] = regressor.fit(time_scaled, Q_latent)

    return surrogate_param


def rollout(surrogate_param, Q_initial, time):

    time_scaled = ((time - surrogate_param['time_start']) / surrogate_param['time_scale'])[:, None]

    Q_latent = surrogate_param['regressor'].predict(time_scaled)

    return pod_func.decompress(surrogate_param['pod_param'], Q_latent)
