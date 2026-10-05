import numpy as np
import pytest

from simwatchdog.io import input_read_func

GAMMA = 1.4


def prim2cons(rho, vel, press):

    return np.array([rho, rho * vel, press / (GAMMA - 1.0) + 0.5 * rho * vel**2])


@pytest.fixture
def advecting_case():

    # a smooth density bump carried at constant velocity and pressure, an exact solution of the Euler equations
    num_cell, num_step = 80, 60

    x    = (np.arange(num_cell) + 0.5) / num_cell
    time = 0.005 * np.arange(num_step)

    Q = np.stack([prim2cons(1.0 + 0.3 * np.exp(-((x - 0.3 - 0.5 * t) / 0.08) ** 2), np.full(num_cell, 0.5), np.ones(num_cell))
                  for t in time])

    meta = {'gamma': GAMMA, 'dx': 1.0 / num_cell, 'x': x, 'num_cell': num_cell, 'num_var': 3, 'time': time}

    return Q, meta


def make_config(**sections):

    return input_read_func.init_config({'case': {'fom_dir': 'unused'}, **sections})
