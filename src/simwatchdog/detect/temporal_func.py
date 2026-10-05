import numpy as np

# Temporal check: CUSUM on the score of another detector.
# It accumulates small but persistent departures from the normal level of that score,
# so a slow drift is flagged before the score itself reaches its threshold.


def init_cusum(baseline_scores, slack=0.5):

    cusum_param = {}

    # normal level and spread of the score over trusted data
    cusum_param['mean']  = np.mean(baseline_scores)
    cusum_param['std']   = max(np.std(baseline_scores), 1e-12)
    cusum_param['slack'] = slack
    cusum_param['sum']   = 0.0

    return cusum_param


def cusum_check(cusum_param, score):

    # departures below the slack (in standard deviations) are forgotten, larger ones add up
    z = (score - cusum_param['mean']) / cusum_param['std']

    cusum_param['sum'] = max(0.0, cusum_param['sum'] + z - cusum_param['slack'])

    return cusum_param['sum'], cusum_param
