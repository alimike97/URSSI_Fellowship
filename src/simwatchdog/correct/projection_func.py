import numpy as np

from simwatchdog.detect import physics_func

# Constraint projection: the smallest changes that make a snapshot physically admissible (1D Euler, ideal gas).


def enforce_positivity(Q, meta, relative_floor=1e-3):

    gamma = meta['gamma']
    Q     = Q.copy()

    # floors relative to the typical density and pressure of the snapshot
    rho_floor = relative_floor * np.mean(np.abs(Q[0]))

    # a cell without mass cannot carry momentum, otherwise its velocity would be unbounded
    no_mass       = Q[0] < rho_floor
    Q[0, no_mass] = rho_floor
    Q[1, no_mass] = 0.0

    rho, vel, press = physics_func.cons2prim(Q, gamma)

    # rebuild the total energy where the pressure is not positive
    press = np.maximum(press, relative_floor * np.mean(np.abs(press)))
    Q[2]  = press / (gamma - 1.0) + 0.5 * rho * vel**2

    return Q


def enforce_conservation(Q, expected, meta):

    # a uniform shift is the smallest (least squares) change that restores the integral of each variable
    domain_length = meta['dx'] * Q.shape[1]
    total         = meta['dx'] * np.sum(Q, axis=1)

    return Q + ((expected - total) / domain_length)[:, None]
