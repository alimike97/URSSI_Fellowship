import numpy as np

# Reference-free physics checks for the 1D Euler equations (ideal gas).
# Q is a conservative snapshot of shape (n_var, n_cell): density, momentum, total energy.
# Every check returns a scalar score, where 0 means nothing suspicious and larger is worse.


def cons2prim(Q, gamma):

    rho   = Q[0]
    vel   = Q[1] / rho
    press = (gamma - 1.0) * (Q[2] - 0.5 * Q[1] * vel)

    return rho, vel, press


def finite_check(Q):

    # fraction of entries that are NaN or inf
    score = np.mean(~np.isfinite(Q))

    return score


def positivity_check(Q, meta):

    with np.errstate(divide='ignore', invalid='ignore'):
        rho, _, press = cons2prim(Q, meta['gamma'])

    # most negative density or pressure, relative to its mean magnitude
    rho_min   = np.nanmin(rho)   / np.nanmean(np.abs(rho))
    press_min = np.nanmin(press) / np.nanmean(np.abs(press))

    score = max(0.0, -rho_min, -press_min)

    return score


def boundary_flux(Q, gamma):

    # net flux entering the domain through the two ends, from the boundary cells
    Q_b = Q[:, [0, -1]]

    _, vel, press = cons2prim(Q_b, gamma)

    flux = np.array([Q_b[1], Q_b[1] * vel + press, (Q_b[2] + press) * vel])

    return flux[:, 0] - flux[:, 1]


def init_conservation(Q, time, meta):

    rho, _, press = cons2prim(Q, meta['gamma'])
    dx            = meta['dx']

    cons_param = {}

    # expected integral of each conserved variable, updated with the boundary fluxes
    cons_param['expected'] = dx * np.sum(Q, axis=1)
    cons_param['net_flux'] = boundary_flux(Q, meta['gamma'])
    cons_param['time']     = time

    # momentum can be zero in total, so it is scaled with density * sound speed
    cons_param['scale']    = dx * np.array([np.sum(rho),
                                            np.sum(np.sqrt(gamma_p_rho(meta['gamma'], press, rho))),
                                            np.sum(np.abs(Q[2]))])

    return cons_param


def gamma_p_rho(gamma, press, rho):

    # (rho * sound speed)^2
    return gamma * press * rho


def conservation_check(cons_param, Q, time, meta):

    net_flux = boundary_flux(Q, meta['gamma'])

    # trapezoidal update of what the totals should be now
    cons_param['expected'] = cons_param['expected'] + 0.5 * (time - cons_param['time']) * (cons_param['net_flux'] + net_flux)
    cons_param['net_flux'] = net_flux
    cons_param['time']     = time

    total = meta['dx'] * np.sum(Q, axis=1)

    # largest relative drift among mass, momentum and energy
    score = np.max(np.abs(total - cons_param['expected']) / cons_param['scale'])

    return score, cons_param


def total_variation(Q):

    return np.sum(np.abs(np.diff(Q, axis=-1)), axis=-1)


def init_total_variation(Q_train):

    # largest total variation of each variable seen in the training snapshots (n_step, n_var, n_cell)
    tv_param = {}

    tv_param['tv_ref'] = np.max(total_variation(Q_train), axis=0)

    # a variable that is flat in training (e.g. momentum at rest) is scaled by its magnitude instead,
    # and skipped if it is zero everywhere
    scale = np.maximum(tv_param['tv_ref'], np.max(np.abs(Q_train), axis=(0, 2)))

    tv_param['scale'] = np.where(scale > 0, scale, np.inf)

    return tv_param


def total_variation_check(tv_param, Q):

    # relative growth of total variation beyond the training data, a sign of spurious oscillations
    growth = (total_variation(Q) - tv_param['tv_ref']) / tv_param['scale']

    score = max(0.0, np.max(growth))

    return score
