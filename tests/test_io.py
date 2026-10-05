import os
import numpy as np
import pytest

from simwatchdog.io import input_read_func
from simwatchdog.io import snapshot_read_func

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write_fake_results(results_dir, iters, num_var=3, num_cell=20):

    os.makedirs(results_dir, exist_ok=True)

    for iter in iters:
        Q = np.full((num_var, num_cell), float(iter))
        np.save(os.path.join(results_dir, f"{iter}iteration_cons.npy"), Q)
        np.save(os.path.join(results_dir, f"{iter}iteration_prim.npy"), Q)
        np.save(os.path.join(results_dir, f"{iter}iteration_res.npy") , Q)


def test_benchmark_input_file_is_valid():

    path = os.path.join(REPO_DIR, 'benchmarks', 'sod_shock_tube', 'input_file.toml')

    config = input_read_func.init_config(input_read_func.read_input_file(path))

    assert config['case']['name'] == 'sod_shock_tube'
    assert config['scoring']['error_threshold'] == 0.05


def test_defaults_are_filled():

    config = input_read_func.init_config({'case': {'fom_dir': 'some/dir'}})

    assert config['scoring']['error_norm'] == 'l1'
    assert config['scoring']['error_variable'] == 'worst'
    assert config['observe']['obs_indices'] == 'all'


def test_missing_fom_dir_fails():

    with pytest.raises(KeyError):
        input_read_func.init_config({'case': {'name': 'no_data'}})


def test_unknown_keyword_fails():

    with pytest.raises(KeyError):
        input_read_func.init_config({'case': {'fom_dir': 'some/dir', 'typo_keyword': 1}})


def test_invalid_option_fails():

    with pytest.raises(ValueError):
        input_read_func.init_config({'case': {'fom_dir': 'some/dir'}, 'scoring': {'error_norm': 'l3'}})


def test_iterations_are_sorted_numerically(tmp_path):

    write_fake_results(tmp_path, [0, 100, 20, 1000])

    assert list(snapshot_read_func.list_iterations(tmp_path)) == [0, 20, 100, 1000]


def test_read_trajectory_shape_and_order(tmp_path):

    write_fake_results(tmp_path, range(0, 101, 10))

    Q, iters = snapshot_read_func.read_trajectory(tmp_path, 'cons', start_iter=0, step_iter=20, end_iter=100)

    assert Q.shape == (6, 3, 20)
    assert list(iters) == [0, 20, 40, 60, 80, 100]
    assert np.all(Q[:, 0, 0] == iters)


def test_rom_layout_with_subfolders(tmp_path):

    write_fake_results(os.path.join(tmp_path, 'cons_prim'), [0, 10])

    Q, _ = snapshot_read_func.read_trajectory(tmp_path)

    assert Q.shape == (2, 3, 20)


def test_missing_results_dir_fails(tmp_path):

    with pytest.raises(FileNotFoundError):
        snapshot_read_func.read_trajectory(os.path.join(tmp_path, 'not_here'))


def test_read_case_meta(tmp_path):

    write_fake_results(tmp_path, range(0, 51, 10))

    config = input_read_func.init_config({'case': {'fom_dir': str(tmp_path), 'step_iter': 10, 'dt': 0.5}})

    Q, meta = snapshot_read_func.read_case(config)

    assert Q.shape == (6, 3, 20)
    assert meta['num_cell'] == 20
    assert np.isclose(meta['x'][0], 0.025) and np.isclose(meta['x'][-1], 0.975)
    assert np.allclose(meta['time'], meta['iters'] * 0.5)


def test_gathered_file_layout(tmp_path):

    # (n_var, n_cell, n_step), saved every 10 iterations and multiplied by the cell width
    num_cell, num_step = 20, 6
    Q_gathered = np.ones((3, num_cell, num_step)) * np.arange(num_step) / num_cell
    np.save(os.path.join(tmp_path, 'cons_gathered.npy'), Q_gathered)

    config = input_read_func.init_config({'case': {'fom_dir': str(tmp_path), 'save_interval': 10, 'step_iter': 20,
                                                   'cons_volume_weighted': True}})

    Q, meta = snapshot_read_func.read_case(config)

    assert Q.shape == (3, 3, num_cell)
    assert list(meta['iters']) == [0, 20, 40]
    assert np.allclose(Q[:, 0, 0], [0, 2, 4])
