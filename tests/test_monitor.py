import numpy as np
import pytest

from simwatchdog.monitor import monitor_func
from simwatchdog.detect import temporal_func
from simwatchdog.correct import assimilation_func, projection_func, memory_func
from simwatchdog.detect import physics_func

from conftest import make_config

NUM_TRAIN = 30


def run_monitor(config, Q, meta, obs_trajectory=None):

    monitor_param = monitor_func.init_monitor(config, Q[:NUM_TRAIN], meta)

    reports = []

    for step in range(Q.shape[0]):
        obs = None if obs_trajectory is None else obs_trajectory[step][:, monitor_param['obs_indices']]
        reports.append(monitor_func.check_snapshot(monitor_param, Q[step], meta['time'][step], obs))

    return reports, monitor_param


def test_clean_training_data_is_not_flagged(advecting_case):

    Q, meta = advecting_case
    config  = make_config(detect={'detectors': ['finite', 'positivity', 'conservation', 'total_variation',
                                                'pod_projection', 'mahalanobis', 'lof'], 'num_modes': 5})

    reports, _ = run_monitor(config, Q[:NUM_TRAIN], meta)

    assert not any(report['flagged'] for report in reports)


def test_out_of_distribution_is_flagged(advecting_case):

    # the bump keeps moving into a region never seen in training
    Q, meta = advecting_case
    config  = make_config(detect={'detectors': ['pod_projection', 'mahalanobis', 'lof'], 'num_modes': 5})

    reports, _ = run_monitor(config, Q, meta)

    for name in ['pod_projection', 'mahalanobis', 'lof']:
        assert reports[-1]['flags'][name]


def test_nonphysical_snapshot_is_flagged(advecting_case):

    Q, meta = advecting_case
    config  = make_config(detect={'detectors': ['finite', 'positivity', 'total_variation']})

    monitor_param = monitor_func.init_monitor(config, Q[:NUM_TRAIN], meta)

    Q_bad        = Q[10].copy()
    Q_bad[2, 5]  = -1.0
    Q_bad[0]    += 0.2 * (-1.0) ** np.arange(Q.shape[2])

    report = monitor_func.check_snapshot(monitor_param, Q_bad, 0.0)

    assert report['flags']['positivity'] and report['flags']['total_variation'] and not report['flags']['finite']

    Q_bad[0, 0] = np.nan
    report      = monitor_func.check_snapshot(monitor_param, Q_bad, 0.0)

    assert all(report['flags'].values())


def test_calibration_raises_thresholds(advecting_case):

    Q, meta = advecting_case
    config  = make_config(detect={'detectors': ['total_variation']})

    # a model whose normal output is slightly oscillatory
    Q_model = Q[:NUM_TRAIN].copy()
    Q_model[:, 0] += 0.02 * (-1.0) ** np.arange(Q.shape[2])

    monitor_param = monitor_func.init_monitor(config, Q[:NUM_TRAIN], meta)
    assert monitor_func.check_snapshot(monitor_param, Q_model[0], 0.0)['flagged']

    monitor_param = monitor_func.calibrate_monitor(monitor_param, Q_model, meta['time'][:NUM_TRAIN])
    assert not monitor_func.check_snapshot(monitor_param, Q_model[0], 0.0)['flagged']


def test_user_threshold_is_kept(advecting_case):

    Q, meta = advecting_case
    config  = make_config(detect={'detectors': ['total_variation'], 'thresholds': {'total_variation': 0.5}})

    monitor_param = monitor_func.init_monitor(config, Q[:NUM_TRAIN], meta)
    monitor_param = monitor_func.calibrate_monitor(monitor_param, Q[:NUM_TRAIN], meta['time'][:NUM_TRAIN])

    assert monitor_param['thresholds']['total_variation'] == 0.5


def test_unknown_detector_fails(advecting_case):

    Q, meta = advecting_case

    with pytest.raises(ValueError):
        monitor_func.init_monitor(make_config(detect={'detectors': ['magic']}), Q, meta)

    with pytest.raises(ValueError):
        monitor_func.init_monitor(make_config(detect={'detectors': ['conservation_cusum']}), Q, meta)


def test_innovation_detects_a_shifted_solution(advecting_case):

    # a solution that looks physical but is in the wrong place
    Q, meta = advecting_case
    config  = make_config(detect={'detectors': ['positivity', 'innovation']}, observe={'obs_step': 4})

    reports, _ = run_monitor(config, np.roll(Q, 10, axis=2), meta, obs_trajectory=Q)

    assert reports[5]['flags']['innovation'] and not reports[5]['flags']['positivity']

    reports, _ = run_monitor(config, Q, meta, obs_trajectory=Q)

    assert not any(report['flagged'] for report in reports)


def test_cusum_flags_a_slow_drift_before_the_threshold():

    rng      = np.random.default_rng(0)
    baseline = 1.0 + 0.1 * rng.standard_normal(200)

    cusum_param = temporal_func.init_cusum(baseline)

    # the score creeps up by one standard deviation, far below a threshold of 1.5 x the largest baseline score
    sums = []
    for score in 1.1 + 0.1 * rng.standard_normal(100):
        cusum_sum, cusum_param = temporal_func.cusum_check(cusum_param, score)
        sums.append(cusum_sum)

    assert max(sums) > 5.0


def test_cusum_detector_in_monitor(advecting_case):

    Q, meta = advecting_case
    config  = make_config(detect={'detectors': ['mahalanobis', 'mahalanobis_cusum'], 'num_modes': 5})

    monitor_param = monitor_func.init_monitor(config, Q[:NUM_TRAIN], meta)
    monitor_param = monitor_func.calibrate_monitor(monitor_param, Q[:NUM_TRAIN], meta['time'][:NUM_TRAIN])

    reports = [monitor_func.check_snapshot(monitor_param, Q[step], meta['time'][step]) for step in range(Q.shape[0])]

    assert not any(report['flags']['mahalanobis_cusum'] for report in reports[:NUM_TRAIN])
    assert reports[-1]['flags']['mahalanobis_cusum']


def test_enkf_matches_kalman_gain():

    rng = np.random.default_rng(0)

    num_member, num_cell, obs_std = 6, 30, 0.3
    obs_indices = np.arange(0, num_cell, 3)

    ensemble = rng.normal(size=(num_member, 1, num_cell))
    obs      = rng.normal(size=(1, obs_indices.size))

    analysis = assimilation_func.enkf_analysis(ensemble, obs, obs_indices, obs_std)

    X = ensemble.reshape(num_member, -1)
    A = X - X.mean(axis=0)
    P = A.T @ A / (num_member - 1)
    H = np.zeros((obs_indices.size, num_cell))
    H[np.arange(obs_indices.size), obs_indices] = 1.0
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + obs_std**2 * np.eye(obs_indices.size))

    assert np.allclose(analysis.ravel(), X.mean(axis=0) + K @ (obs.ravel() - H @ X.mean(axis=0)))


def test_memory_finds_the_closest_snapshots(advecting_case):

    Q, _        = advecting_case
    obs_indices = np.arange(0, Q.shape[2], 4)

    memory_param = memory_func.init_memory(Q, obs_indices, num_neighbors=3)
    neighbors    = memory_func.find_neighbors(memory_param, Q[20][:, obs_indices])

    assert np.array_equal(neighbors[0], Q[20])
    assert neighbors.shape == (3, 3, Q.shape[2])


def test_memory_correction_recovers_the_truth(advecting_case):

    Q, meta = advecting_case
    config  = make_config(detect={'detectors': ['innovation']}, observe={'obs_step': 4},
                          correct={'method': 'memory', 'num_neighbors': 6})

    monitor_param = monitor_func.init_monitor(config, Q[:NUM_TRAIN], meta)

    # the library holds every second snapshot, the truth to recover is one that was left out
    monitor_param = monitor_func.init_correction(monitor_param, config, Q[::2])

    truth  = Q[41]
    obs    = truth[:, monitor_param['obs_indices']]
    Q_bad  = np.roll(truth, 15, axis=1)

    report              = monitor_func.check_snapshot(monitor_param, Q_bad, 0.0, obs)
    Q_corrected, report = monitor_func.correct_snapshot(monitor_param, Q_bad, report, obs)

    def error(Q):
        return np.linalg.norm(Q - truth) / np.linalg.norm(truth)

    assert report['corrected']
    assert error(Q_corrected) < 0.1 * error(Q_bad)


def test_unflagged_snapshot_is_not_touched(advecting_case):

    Q, meta = advecting_case
    config  = make_config(detect={'detectors': ['positivity']}, correct={'method': 'projection'})

    monitor_param = monitor_func.init_monitor(config, Q[:NUM_TRAIN], meta)
    monitor_param = monitor_func.init_correction(monitor_param, config)

    report              = monitor_func.check_snapshot(monitor_param, Q[5], 0.0)
    Q_corrected, report = monitor_func.correct_snapshot(monitor_param, Q[5], report)

    assert Q_corrected is Q[5] or np.array_equal(Q_corrected, Q[5])
    assert not report['corrected']


def test_projection_restores_admissibility(advecting_case):

    Q, meta = advecting_case

    Q_bad        = Q[5].copy()
    Q_bad[0, 3]  = -0.2
    Q_bad[2, 40] = -1.0

    Q_fixed = projection_func.enforce_positivity(Q_bad, meta)

    assert physics_func.positivity_check(Q_fixed, meta) == 0
    assert np.all(np.isfinite(Q_fixed)) and np.max(np.abs(Q_fixed)) < 10

    expected = meta['dx'] * np.sum(Q[5], axis=1)
    Q_fixed  = projection_func.enforce_conservation(Q[5] * 0.9, expected, meta)

    assert np.allclose(meta['dx'] * np.sum(Q_fixed, axis=1), expected)
