import numpy as np

from simwatchdog.detect import physics_func

GAMMA = 1.4


def prim2cons(rho, vel, press):

    return np.array([rho, rho * vel, press / (GAMMA - 1.0) + 0.5 * rho * vel**2])


def sod_like_state(num_cell=100):

    x     = (np.arange(num_cell) + 0.5) / num_cell
    rho   = np.where(x < 0.5, 1.0, 0.125)
    press = np.where(x < 0.5, 1.0, 0.1)

    meta = {'gamma': GAMMA, 'dx': 1.0 / num_cell}

    return prim2cons(rho, np.zeros(num_cell), press), meta


def test_cons2prim_round_trip():

    Q = prim2cons(np.array([1.0, 0.5]), np.array([0.3, -0.2]), np.array([1.0, 0.4]))

    rho, vel, press = physics_func.cons2prim(Q, GAMMA)

    assert np.allclose(rho, [1.0, 0.5]) and np.allclose(vel, [0.3, -0.2]) and np.allclose(press, [1.0, 0.4])


def test_finite_check():

    Q, _ = sod_like_state()
    assert physics_func.finite_check(Q) == 0

    Q[0, 3] = np.nan
    assert physics_func.finite_check(Q) > 0


def test_positivity_check():

    Q, meta = sod_like_state()
    assert physics_func.positivity_check(Q, meta) == 0

    # remove energy from one cell until its pressure is negative
    Q[2, 10] = -0.1
    assert physics_func.positivity_check(Q, meta) > 0


def test_conservation_holds_with_boundary_fluxes():

    # a gas at rest with different end pressures is steady in mass and energy,
    # the momentum change is fully explained by the boundary pressure difference
    Q, meta = sod_like_state()

    cons_param = physics_func.init_conservation(Q, 0.0, meta)

    Q_new     = Q.copy()
    Q_new[1] += 0.1 * (1.0 - 0.1)

    score, cons_param = physics_func.conservation_check(cons_param, Q_new, 0.1, meta)

    # the boundary cells now carry momentum so the flux estimate is not exact, but the drift stays small
    assert score < 0.02


def test_conservation_detects_mass_loss():

    Q, meta = sod_like_state()

    cons_param = physics_func.init_conservation(Q, 0.0, meta)

    Q_new       = Q.copy()
    Q_new[0]   *= 0.95
    Q_new[2]   *= 0.95

    score, _ = physics_func.conservation_check(cons_param, Q_new, 0.0, meta)

    assert np.isclose(score, 0.05)


def test_total_variation_detects_oscillations():

    Q, _ = sod_like_state()

    tv_param = physics_func.init_total_variation(Q[None])

    assert physics_func.total_variation_check(tv_param, Q) == 0

    Q_osc     = Q.copy()
    Q_osc[0] += 0.05 * (-1.0) ** np.arange(Q.shape[1])

    assert physics_func.total_variation_check(tv_param, Q_osc) > 1.0
