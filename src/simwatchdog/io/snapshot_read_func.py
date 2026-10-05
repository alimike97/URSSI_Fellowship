import os
import re
import numpy as np

# CompFlowLab saves one file per saved iteration: <iter>iteration_<kind>.npy
# kind is cons, prim or res and each file has the shape (n_var, n_cell)
# alternatively the whole run can be gathered in one file: <kind>_gathered.npy


def find_snapshot_dir(results_dir):

    # the FOM solver saves directly in the results folder, the ROM solvers in a cons_prim subfolder
    sub_dir = os.path.join(results_dir, 'cons_prim')

    if os.path.isdir(sub_dir):
        return sub_dir

    if os.path.isdir(results_dir):
        return results_dir

    raise FileNotFoundError(f"The results directory '{results_dir}' does not exist.")


def list_iterations(results_dir, kind='cons'):

    snapshot_dir = find_snapshot_dir(results_dir)
    pattern      = re.compile(rf"^(\d+)iteration_{kind}\.npy$")

    iters = [int(m.group(1)) for m in map(pattern.match, os.listdir(snapshot_dir)) if m]

    return np.array(sorted(iters), dtype=int)


def read_snapshot(results_dir, iter, kind='cons'):

    snapshot_dir = find_snapshot_dir(results_dir)

    # residuals of the ROM solvers are kept in their own subfolder
    if kind == 'res' and os.path.isdir(os.path.join(results_dir, 'res')):
        snapshot_dir = os.path.join(results_dir, 'res')

    return np.load(os.path.join(snapshot_dir, f"{iter}iteration_{kind}.npy"))


def read_gathered(results_dir, kind='cons', save_interval=1):

    # a whole trajectory gathered in one file of shape (n_var, n_cell, n_step)
    Q = np.load(os.path.join(results_dir, f"{kind}_gathered.npy"))
    Q = np.transpose(Q, (2, 0, 1))

    available = np.arange(Q.shape[0]) * save_interval

    return Q, available


def read_trajectory(results_dir, kind='cons', start_iter=0, step_iter=1, end_iter=None, save_interval=1):

    is_gathered = os.path.isfile(os.path.join(results_dir, f"{kind}_gathered.npy"))

    if is_gathered:
        Q_all, available = read_gathered(results_dir, kind, save_interval)
    else:
        available = list_iterations(results_dir, 'cons')

    if available.size == 0:
        raise FileNotFoundError(f"No snapshots found in '{results_dir}'.")

    if end_iter is None:
        end_iter = available[-1]

    requested = np.arange(start_iter, end_iter + 1, step_iter)
    iters     = np.intersect1d(requested, available)

    if iters.size == 0:
        raise ValueError(f"None of the requested iterations ({start_iter}:{step_iter}:{end_iter}) are saved in '{results_dir}'.")

    if iters.size < requested.size:
        print(f"Warning: {requested.size - iters.size} of {requested.size} requested iterations are not saved and were skipped.")

    # (n_step, n_var, n_cell)
    if is_gathered:
        Q = Q_all[np.searchsorted(available, iters)]
    else:
        Q = np.stack([read_snapshot(results_dir, iter, kind) for iter in iters])

    return Q, iters


def init_meta(config, Q):

    case     = config['case']
    num_cell = Q.shape[-1]
    dx       = (case['x_final'] - case['x_initial']) / num_cell

    meta = {}

    meta['num_var']   = Q.shape[-2]
    meta['num_cell']  = num_cell
    meta['dx']        = dx
    meta['x']         = case['x_initial'] + (np.arange(num_cell) + 0.5) * dx
    meta['dt']        = case['dt']
    meta['gamma']     = case['gamma']
    meta['var_names'] = case['cons_var_names']

    return meta


def read_case(config):

    case = config['case']

    Q, iters = read_trajectory(case['fom_dir'], 'cons', case['start_iter'], case['step_iter'], case['end_iter'],
                               case['save_interval'])

    meta = init_meta(config, Q)

    # CompFlowLab stores the conservative variables multiplied by the cell width
    if case['cons_volume_weighted']:
        Q = Q / meta['dx']

    meta['iters'] = iters
    meta['time']  = iters * case['dt']

    return Q, meta
