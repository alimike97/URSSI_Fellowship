import numpy as np

from simwatchdog.detect import physics_func
from simwatchdog.detect import ood_func
from simwatchdog.detect import residual_func
from simwatchdog.detect import temporal_func
from simwatchdog.detect import innovation_func
from simwatchdog.correct import projection_func
from simwatchdog.correct import memory_func
from simwatchdog.correct import assimilation_func

# The monitor is the public interface of the package:
#   monitor_param = init_monitor(config, Q_train, meta)
#   (meta['time'] holds the times of Q_train; a solver can pass its own flux divergence for the residual detector)
#   monitor_param = calibrate_monitor(monitor_param, Q_baseline, time_baseline)   (optional)
#   monitor_param = init_correction(monitor_param, config, Q_library)             (optional)
#   report        = check_snapshot(monitor_param, Q, time)
#   Q, report     = correct_snapshot(monitor_param, Q, report)
#
# obs is the observed true state at the cells monitor_param['obs_indices'], shape (n_var, n_obs).
# It is only needed by the innovation detector and the memory correction.

PHYSICS_DETECTORS = ['finite', 'positivity', 'conservation', 'total_variation', 'residual']
OOD_DETECTORS     = ['pod_projection', 'mahalanobis', 'lof']
OBS_DETECTORS     = ['innovation']
CUSUM_SUFFIX      = '_cusum'

# a snapshot is flagged when the score is above the threshold
DEFAULT_THRESHOLDS = {
    'finite'          : 0.0,
    'positivity'      : 0.0,
    'conservation'    : 1e-3,
    'total_variation' : 0.05,
    'innovation'      : 0.02,
    'cusum'           : 5.0,
}


def init_obs_indices(config, num_cell):

    observe = config['observe']

    if observe['obs_indices'] == 'all':
        return np.arange(0, num_cell, observe['obs_step'])

    return np.array(observe['obs_indices'], dtype=int)


def ood_score(monitor_param, name, Q):

    ood_param = monitor_param['ood_param']

    if name == 'pod_projection':
        return ood_func.pod_projection_check(ood_param, Q)

    if name == 'mahalanobis':
        return ood_func.mahalanobis_check(ood_param, Q)

    if name == 'lof':
        return ood_func.lof_check(ood_param, Q)


def reset_residual(monitor_param):

    # the residual forgets the last snapshot it has seen
    if 'residual_param' in monitor_param:
        for key in ['Q_previous', 'flux_divergence', 'time']:
            monitor_param['residual_param'].pop(key, None)


def init_monitor(config, Q_train, meta, flux_divergence_func=None):

    detectors = config['detect']['detectors']
    base      = PHYSICS_DETECTORS + OOD_DETECTORS + OBS_DETECTORS

    for name in detectors:

        if name.endswith(CUSUM_SUFFIX):
            if name.removesuffix(CUSUM_SUFFIX) not in detectors:
                raise ValueError(f"'{name}' needs '{name.removesuffix(CUSUM_SUFFIX)}' in the list of detectors.")

        elif name not in base:
            raise ValueError(f"Unknown detector '{name}'. Options: {base}, or one of them followed by {CUSUM_SUFFIX}")

    monitor_param = {}

    # the cusum detectors come last since they use the scores of the others
    monitor_param['detectors']          = sorted(detectors, key=lambda name: name.endswith(CUSUM_SUFFIX))
    monitor_param['calibration_factor'] = config['detect']['calibration_factor']
    monitor_param['user_thresholds']    = config['detect']['thresholds']
    monitor_param['meta']               = meta
    monitor_param['obs_indices']        = init_obs_indices(config, meta['num_cell'])
    monitor_param['cusum_param']        = {}
    monitor_param['thresholds']         = {}

    for name in detectors:

        if name.endswith(CUSUM_SUFFIX):
            monitor_param['thresholds'][name] = DEFAULT_THRESHOLDS['cusum']

        elif name in DEFAULT_THRESHOLDS:
            monitor_param['thresholds'][name] = DEFAULT_THRESHOLDS[name]

    if 'total_variation' in detectors:
        monitor_param['tv_param'] = physics_func.init_total_variation(Q_train)

    if 'residual' in detectors:

        time_train = meta['time'][:Q_train.shape[0]]

        monitor_param['residual_param'] = residual_func.init_residual(Q_train, time_train, flux_divergence_func)

        # the built-in flux is not the scheme of the solver, so even the training data has a residual
        # that depends on the grid. The threshold is set from it.
        train_scores = []

        for Q, time in zip(Q_train, time_train):
            score, monitor_param['residual_param'] = residual_func.residual_check(monitor_param['residual_param'], Q, time, meta)
            train_scores.append(score)

        monitor_param['thresholds']['residual'] = monitor_param['calibration_factor'] * max(train_scores)

        reset_residual(monitor_param)

    if 'innovation' in detectors:
        monitor_param['innovation_param'] = innovation_func.init_innovation(Q_train, monitor_param['obs_indices'])

    if any(name in OOD_DETECTORS for name in detectors):

        monitor_param['ood_param'] = ood_func.init_ood(Q_train, config['detect']['num_modes'])

        # out-of-distribution scores have no natural scale, so their thresholds come from the training data
        for name in detectors:
            if name in OOD_DETECTORS:
                train_scores = [ood_score(monitor_param, name, Q) for Q in Q_train]
                monitor_param['thresholds'][name] = monitor_param['calibration_factor'] * max(train_scores)

    # thresholds given in the input file override the defaults
    monitor_param['thresholds'].update(monitor_param['user_thresholds'])

    return monitor_param


def calibrate_monitor(monitor_param, Q_baseline, time_baseline, obs_baseline=None):

    # Q_baseline is a stretch of the model output that is known to be acceptable, e.g. its reproduction
    # of the training window. Thresholds are raised so that this normal behavior is not flagged.
    if obs_baseline is None:
        obs_baseline = [None] * len(Q_baseline)

    reports = [check_snapshot(monitor_param, Q, time, obs) for Q, time, obs in zip(Q_baseline, time_baseline, obs_baseline)]

    for name in monitor_param['detectors']:

        if name.endswith(CUSUM_SUFFIX):

            # the cusum detectors learn the normal level of the score they follow
            baseline_scores = [report['scores'][name.removesuffix(CUSUM_SUFFIX)] for report in reports]

            if np.all(np.isfinite(baseline_scores)):

                cusum_param   = temporal_func.init_cusum(baseline_scores)
                baseline_sums = []

                # the score can drift within the baseline itself, which should not be flagged either
                for score in baseline_scores:
                    cusum_sum, cusum_param = temporal_func.cusum_check(cusum_param, score)
                    baseline_sums.append(cusum_sum)

                cusum_param['sum'] = 0.0

                monitor_param['cusum_param'][name] = cusum_param

                if name not in monitor_param['user_thresholds']:
                    monitor_param['thresholds'][name] = max(monitor_param['thresholds'][name],
                                                            monitor_param['calibration_factor'] * max(baseline_sums))

            continue

        if name in monitor_param['user_thresholds']:
            continue

        baseline = max(report['scores'][name] for report in reports)

        if np.isfinite(baseline):
            monitor_param['thresholds'][name] = max(monitor_param['thresholds'][name],
                                                    monitor_param['calibration_factor'] * baseline)

    # the conservation reference is taken again from the first monitored snapshot
    monitor_param.pop('cons_param', None)

    reset_residual(monitor_param)

    return monitor_param


def check_snapshot(monitor_param, Q, time, obs=None):

    meta   = monitor_param['meta']
    scores = {}

    is_finite = bool(np.all(np.isfinite(Q)))

    for name in monitor_param['detectors']:

        if name == 'finite':
            scores[name] = physics_func.finite_check(Q)

        # a snapshot with NaN or inf cannot be evaluated, which is the worst possible score
        elif not is_finite:
            scores[name] = np.inf

        elif name == 'positivity':
            scores[name] = physics_func.positivity_check(Q, meta)

        elif name == 'conservation':

            # the first snapshot seen by the monitor is the reference
            if 'cons_param' not in monitor_param:
                monitor_param['cons_param'] = physics_func.init_conservation(Q, time, meta)
                scores[name] = 0.0
            else:
                scores[name], monitor_param['cons_param'] = physics_func.conservation_check(monitor_param['cons_param'], Q, time, meta)

        elif name == 'residual':
            scores[name], monitor_param['residual_param'] = residual_func.residual_check(monitor_param['residual_param'], Q, time, meta)

        elif name == 'total_variation':
            scores[name] = physics_func.total_variation_check(monitor_param['tv_param'], Q)

        elif name == 'innovation':

            if obs is None:
                raise ValueError("The innovation detector needs observations: check_snapshot(monitor_param, Q, time, obs)")

            scores[name] = innovation_func.innovation_check(monitor_param['innovation_param'], Q, obs)

        elif name.endswith(CUSUM_SUFFIX):

            # silent until the monitor is calibrated
            if name in monitor_param['cusum_param']:
                scores[name], monitor_param['cusum_param'][name] = temporal_func.cusum_check(
                    monitor_param['cusum_param'][name], scores[name.removesuffix(CUSUM_SUFFIX)])
            else:
                scores[name] = 0.0

        else:
            scores[name] = ood_score(monitor_param, name, Q)

    report = {}

    report['time']      = time
    report['scores']    = scores
    report['flags']     = {name: bool(scores[name] > monitor_param['thresholds'][name]) for name in scores}
    report['flagged']   = any(report['flags'].values())
    report['corrected'] = False

    return report


def init_correction(monitor_param, config, Q_library=None):

    correct = config['correct']

    monitor_param['correct_method'] = correct['method']
    monitor_param['obs_std']        = config['observe']['obs_std']

    if correct['method'] == 'memory':

        if Q_library is None:
            raise ValueError("The memory correction needs a library of trusted snapshots.")

        monitor_param['memory_param'] = memory_func.init_memory(Q_library, monitor_param['obs_indices'], correct['num_neighbors'])

    return monitor_param


def correct_snapshot(monitor_param, Q, report, obs=None):

    method = monitor_param.get('correct_method', 'none')
    meta   = monitor_param['meta']

    if method == 'none' or not report['flagged']:
        return Q, report

    if method == 'projection':

        # nothing can be projected if the snapshot has NaN or inf
        if not np.all(np.isfinite(Q)):
            return Q, report

        Q_corrected = projection_func.enforce_positivity(Q, meta)

        if 'cons_param' in monitor_param:
            Q_corrected = projection_func.enforce_conservation(Q_corrected, monitor_param['cons_param']['expected'], meta)

    if method == 'memory':

        if obs is None:
            raise ValueError("The memory correction needs observations: correct_snapshot(monitor_param, Q, report, obs)")

        memory_param = monitor_param['memory_param']
        var_scale    = memory_param['var_scale'][:, None]

        # closest trusted snapshots as the background ensemble, pulled to the observations
        ensemble = memory_func.find_neighbors(memory_param, obs)

        Q_corrected = var_scale * assimilation_func.enkf_analysis(ensemble / var_scale, obs / var_scale,
                                                                  monitor_param['obs_indices'], monitor_param['obs_std'])

        Q_corrected = projection_func.enforce_positivity(Q_corrected, meta)

    report['corrected']  = True
    report['correction'] = method

    return Q_corrected, report
