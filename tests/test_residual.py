import numpy as np

from simwatchdog.detect import residual_func
from simwatchdog.monitor import monitor_func

from conftest import make_config, prim2cons


def run_residual(Q, time, meta, Q_train):

    residual_param = residual_func.init_residual(Q_train, meta['time'])

    scores = []

    for step in range(Q.shape[0]):
        score, residual_param = residual_func.residual_check(residual_param, Q[step], time[step], meta)
        scores.append(score)

    return np.array(scores)


def test_uniform_state_has_no_flux_divergence():

    num_cell = 50
    Q        = prim2cons(np.ones(num_cell), np.full(num_cell, 0.3), np.ones(num_cell))

    divergence = residual_func.euler_flux_divergence(Q, {'gamma': 1.4, 'dx': 1.0 / num_cell})

    assert np.allclose(divergence, 0)


def test_exact_solution_has_a_small_residual(advecting_case):

    Q, meta = advecting_case

    scores = run_residual(Q, meta['time'], meta, Q)

    # not zero: the built-in flux is first order, so the residual of an exact solution scales with the cell width
    assert scores[0] == 0
    assert np.max(scores) < 0.5


def test_wrong_wave_speed_has_a_large_residual(advecting_case):

    # the same bump carried at twice the speed: every snapshot is physical, the motion is not
    Q, meta = advecting_case

    Q_wrong = Q[::2]

    scores_exact = run_residual(Q, meta['time'], meta, Q)
    scores_wrong = run_residual(Q_wrong, meta['time'], meta, Q)

    assert np.mean(scores_wrong[1:]) > 2 * np.mean(scores_exact[1:])


def test_inadmissible_state_is_the_worst_score(advecting_case):

    Q, meta = advecting_case

    Q_bad       = Q[:3].copy()
    Q_bad[1, 2] = -5.0

    assert run_residual(Q_bad, meta['time'], meta, Q)[1] == np.inf


def test_solver_can_give_its_own_flux_divergence(advecting_case):

    Q, meta = advecting_case
    calls   = []

    def flux_divergence_func(Q, meta):
        calls.append(1)
        return residual_func.euler_flux_divergence(Q, meta)

    config        = make_config(detect={'detectors': ['residual']})
    monitor_param = monitor_func.init_monitor(config, Q[:30], meta, flux_divergence_func)

    # already used once per training snapshot to set the threshold
    assert len(calls) == 30

    monitor_func.check_snapshot(monitor_param, Q[0], meta['time'][0])
    report = monitor_func.check_snapshot(monitor_param, Q[1], meta['time'][1])

    assert len(calls) == 32
    assert not report['flags']['residual']
