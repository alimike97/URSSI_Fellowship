import numpy as np

from simwatchdog.detect import physics_func

# Residual check: do two consecutive snapshots satisfy the governing equations, dQ/dt + dF/dx = 0 ?
# The flux divergence dF/dx comes from a small built-in scheme for the 1D Euler equations, or from a
# function given by the solver: flux_divergence_func(Q, meta) -> array of shape (n_var, n_cell).


def euler_flux(Q, gamma):

    rho, vel, press = physics_func.cons2prim(Q, gamma)

    flux        = np.array([Q[1], Q[1] * vel + press, (Q[2] + press) * vel])
    wave_speed  = np.abs(vel) + np.sqrt(gamma * press / rho)

    return flux, wave_speed


def euler_flux_divergence(Q, meta):

    # first order finite volume scheme with the Rusanov (local Lax-Friedrichs) flux, zero gradient at both ends
    Q_ghost = np.concatenate([Q[:, :1], Q, Q[:, -1:]], axis=1)

    with np.errstate(invalid='ignore', divide='ignore'):
        flux, wave_speed = euler_flux(Q_ghost, meta['gamma'])

    Q_left , Q_right  = Q_ghost[:, :-1], Q_ghost[:, 1:]
    speed             = np.maximum(wave_speed[:-1], wave_speed[1:])

    flux_face = 0.5 * (flux[:, :-1] + flux[:, 1:]) - 0.5 * speed * (Q_right - Q_left)

    return (flux_face[:, 1:] - flux_face[:, :-1]) / meta['dx']


def init_residual(Q_train, time_train, flux_divergence_func=None):

    residual_param = {}

    residual_param['flux_divergence_func'] = flux_divergence_func or euler_flux_divergence

    # typical rate of change of each variable, so the score reads as a relative imbalance
    rate = np.abs(np.diff(Q_train, axis=0)) / np.diff(time_train)[:, None, None]

    residual_param['rate_scale'] = np.mean(rate, axis=(0, 2))

    return residual_param


def residual_check(residual_param, Q, time, meta):

    flux_divergence = residual_param['flux_divergence_func'](Q, meta)

    # the first snapshot has no previous one to compare with
    if 'Q_previous' not in residual_param:
        score = 0.0

    else:
        dQ_dt = (Q - residual_param['Q_previous']) / (time - residual_param['time'])

        # the flux divergence is averaged over the two snapshots (trapezoidal rule)
        residual = dQ_dt + 0.5 * (flux_divergence + residual_param['flux_divergence'])

        score = np.max(np.mean(np.abs(residual), axis=1) / residual_param['rate_scale'])

        # an inadmissible state (e.g. negative pressure) has no defined flux
        if not np.isfinite(score):
            score = np.inf

    residual_param['Q_previous']      = Q
    residual_param['flux_divergence'] = flux_divergence
    residual_param['time']            = time

    return score, residual_param
