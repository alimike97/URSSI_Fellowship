import os
import numpy as np
import pytest

from simwatchdog.testbed import pod_func, run_func

from conftest import make_config


def test_pod_round_trip(advecting_case):

    Q, _ = advecting_case

    pod_param = pod_func.init_pod(Q, num_modes=20)
    Q_latent  = pod_func.compress(pod_param, Q)

    assert Q_latent.shape == (Q.shape[0], 20)
    assert np.allclose(pod_func.decompress(pod_param, Q_latent), Q, atol=1e-3)


@pytest.mark.parametrize('method', ['opinf', 'pod_regression'])
def test_benchmark_runs_end_to_end(advecting_case, tmp_path, method):

    if method == 'opinf':
        pytest.importorskip('opinf')

    pytest.importorskip('matplotlib')

    # write the case the way CompFlowLab gathers it: (n_var, n_cell, n_step)
    Q, _ = advecting_case
    np.save(os.path.join(tmp_path, 'cons_gathered.npy'), np.transpose(Q, (1, 2, 0)))

    config = make_config(case={'fom_dir': str(tmp_path), 'dt': 0.005},
                         surrogate={'method': method, 'train_end_iter': 29, 'num_modes': 5},
                         detect={'detectors': ['finite', 'positivity', 'conservation', 'mahalanobis', 'innovation'], 'num_modes': 5},
                         observe={'obs_step': 4},
                         correct={'method': 'memory', 'library_step': 2})

    results = run_func.run(config, str(tmp_path))

    assert results['Q_surrogate'].shape == Q.shape
    assert results['num_train'] == 30

    # the surrogate reproduces the training window
    assert np.max(results['error'][:30]) < 0.05

    # beyond it the state leaves the training data, is flagged and corrected
    assert results['detector_scores']['any']['first_flag'] is not None
    assert np.mean(results['error_corrected'][30:]) < np.mean(results['error'][30:])

    for name in ['summary.json', 'summary.png', 'last_snapshot.png', 'surrogate_cons.npy', 'corrected_cons.npy']:
        assert os.path.isfile(os.path.join(results['dir_results'], name))
