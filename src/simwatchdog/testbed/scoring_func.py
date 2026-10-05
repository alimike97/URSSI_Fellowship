import numpy as np

# Scoring against the FOM. Trajectories have the shape (n_step, n_var, n_cell).

NORM_ORDER = {'l1': 1, 'l2': 2, 'linf': np.inf}


def error_curve(Q_surrogate, Q_fom, config):

    scoring   = config['scoring']
    var_names = config['case']['cons_var_names']
    order     = NORM_ORDER[scoring['error_norm']]

    # error of each variable at each step: (n_step, n_var)
    error = np.linalg.norm(Q_surrogate - Q_fom, ord=order, axis=-1)

    # a variable can be close to zero at some steps (e.g. momentum at rest),
    # so each variable is normalized by its largest FOM norm over the trajectory
    scale = np.max(np.linalg.norm(Q_fom, ord=order, axis=-1), axis=0)
    error = error / scale

    # a blown-up surrogate counts as an infinite error
    error = np.where(np.isfinite(error), error, np.inf)

    if scoring['error_variable'] == 'worst':
        return np.max(error, axis=1)

    return error[:, var_names.index(scoring['error_variable'])]


def failure_onset(error, config):

    # first step where the error against the FOM exceeds the threshold, None if it never does
    failed = np.flatnonzero(error > config['scoring']['error_threshold'])

    if failed.size == 0:
        return None

    return int(failed[0])


def score_detector(flags, onset):

    flags   = np.asarray(flags, dtype=bool)
    flagged = np.flatnonzero(flags)

    score = {}

    score['onset']      = onset
    score['first_flag'] = int(flagged[0]) if flagged.size > 0 else None

    # the surrogate never failed, so every flag is a false alarm
    if onset is None:

        score['detected']     = False
        score['missed']       = False
        score['delay']        = None
        score['false_alarms'] = int(flagged.size)

        return score

    after_onset = flagged[flagged >= onset]

    score['detected']     = after_onset.size > 0
    score['missed']       = after_onset.size == 0
    score['delay']        = int(after_onset[0] - onset) if after_onset.size > 0 else None
    score['false_alarms'] = int(np.sum(flagged < onset))

    return score
