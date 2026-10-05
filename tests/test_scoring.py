import numpy as np

from simwatchdog.io import input_read_func
from simwatchdog.testbed import scoring_func


def make_config(**scoring):

    return input_read_func.init_config({'case': {'fom_dir': 'unused'}, 'scoring': scoring})


def make_trajectories(num_step=50, num_cell=40):

    # FOM is constant in time, the surrogate error in density grows by 0.2 % per step
    Q_fom = np.ones((num_step, 3, num_cell))
    Q_fom[:, 1] = 0.0
    Q_fom[-1, 1] = 1.0

    Q_surrogate = Q_fom.copy()
    Q_surrogate[:, 0] += 0.002 * np.arange(num_step)[:, None]

    return Q_surrogate, Q_fom


def test_error_curve_and_onset():

    Q_surrogate, Q_fom = make_trajectories()
    config             = make_config(error_threshold=0.051)

    error = scoring_func.error_curve(Q_surrogate, Q_fom, config)

    assert error.shape == (50,)
    assert np.allclose(error, 0.002 * np.arange(50))

    # 5.1 % is first exceeded at step 26
    assert scoring_func.failure_onset(error, config) == 26


def test_error_variable_selection():

    Q_surrogate, Q_fom = make_trajectories()

    error = scoring_func.error_curve(Q_surrogate, Q_fom, make_config(error_variable='Total Energy'))

    assert np.all(error == 0)
    assert scoring_func.failure_onset(error, make_config()) is None


def test_blown_up_surrogate_is_a_failure():

    Q_surrogate, Q_fom = make_trajectories()
    Q_surrogate[10:]   = np.nan
    config             = make_config()

    error = scoring_func.error_curve(Q_surrogate, Q_fom, config)

    assert scoring_func.failure_onset(error, config) == 10


def test_score_detector_delay_and_false_alarms():

    flags        = np.zeros(50, dtype=bool)
    flags[[5, 30, 31]] = True

    score = scoring_func.score_detector(flags, onset=26)

    assert score['detected'] and not score['missed']
    assert score['delay'] == 4
    assert score['false_alarms'] == 1


def test_score_detector_miss():

    score = scoring_func.score_detector(np.zeros(50, dtype=bool), onset=26)

    assert score['missed'] and score['delay'] is None and score['false_alarms'] == 0


def test_score_detector_without_failure():

    flags     = np.zeros(50, dtype=bool)
    flags[3]  = True

    score = scoring_func.score_detector(flags, onset=None)

    assert not score['detected'] and not score['missed'] and score['false_alarms'] == 1
