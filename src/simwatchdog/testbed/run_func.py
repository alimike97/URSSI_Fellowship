import os
import json
import importlib
import numpy as np

from simwatchdog.io import snapshot_read_func
from simwatchdog.monitor import monitor_func
from simwatchdog.testbed import scoring_func
from simwatchdog.visual import visual_func

# the surrogate modules are imported on demand so their packages are only needed when used
SURROGATE_MODULES = {
    'opinf'          : 'simwatchdog.testbed.opinf_model',
    'pod_regression' : 'simwatchdog.testbed.pod_regression_model',
    'cnn'            : 'simwatchdog.testbed.cnn_model',
}


def run_surrogate(config, Q_fom, meta):

    surrogate = config['surrogate']
    time      = meta['time']

    # the surrogate only sees the first part of the FOM trajectory
    train_end_iter = surrogate['train_end_iter']

    if train_end_iter is None:
        train_end_iter = meta['iters'][meta['iters'].size // 2]

    num_train = int(np.sum(meta['iters'] <= train_end_iter))

    model = importlib.import_module(SURROGATE_MODULES[surrogate['method']])

    surrogate_param = model.train(config, Q_fom[:num_train], time[:num_train])
    Q_surrogate     = model.rollout(surrogate_param, Q_fom[0], time)

    return Q_surrogate, num_train


def run_monitor(config, Q_fom, Q_surrogate, num_train, meta):

    time    = meta['time']
    Q_train = Q_fom[:num_train]

    monitor_param = monitor_func.init_monitor(config, Q_train, meta)

    # the FOM plays the role of the measurements
    obs = Q_fom[:, :, monitor_param['obs_indices']]

    # the surrogate output over the training window is taken as its normal behavior
    if config['detect']['calibrate']:
        monitor_param = monitor_func.calibrate_monitor(monitor_param, Q_surrogate[:num_train], time[:num_train], obs[:num_train])

    # trusted snapshots for the memory correction
    library_end   = num_train if config['correct']['library'] == 'train' else Q_fom.shape[0]
    Q_library     = Q_fom[:library_end:config['correct']['library_step']]
    monitor_param = monitor_func.init_correction(monitor_param, config, Q_library)

    reports     = []
    Q_corrected = Q_surrogate.copy()

    for step in range(time.size):

        report = monitor_func.check_snapshot(monitor_param, Q_surrogate[step], time[step], obs[step])

        # each snapshot is corrected on its own, the surrogate is not restarted from the corrected state
        Q_corrected[step], report = monitor_func.correct_snapshot(monitor_param, Q_surrogate[step], report, obs[step])

        reports.append(report)

    # (n_step,) arrays for each detector
    scores = {name: np.array([report['scores'][name] for report in reports]) for name in monitor_param['detectors']}
    flags  = {name: np.array([report['flags'][name]  for report in reports]) for name in monitor_param['detectors']}

    return scores, flags, Q_corrected, monitor_param


def run(config, working_dir):

    Q_fom, meta = snapshot_read_func.read_case(config)

    Q_surrogate, num_train = run_surrogate(config, Q_fom, meta)

    scores, flags, Q_corrected, monitor_param = run_monitor(config, Q_fom, Q_surrogate, num_train, meta)

    # ground truth from the FOM
    error           = scoring_func.error_curve(Q_surrogate, Q_fom, config)
    error_corrected = scoring_func.error_curve(Q_corrected, Q_fom, config)
    onset           = scoring_func.failure_onset(error, config)

    detector_scores = {name: scoring_func.score_detector(flags[name], onset) for name in flags}

    # combined result of the detectors that do not use observations, and of all of them
    no_obs = [flags[name] for name in flags if name not in monitor_func.OBS_DETECTORS]

    if len(no_obs) > 0:
        detector_scores['any (no obs)'] = scoring_func.score_detector(np.any(no_obs, axis=0), onset)

    if len(no_obs) < len(flags):
        detector_scores['any'] = scoring_func.score_detector(np.any(list(flags.values()), axis=0), onset)

    results = {}

    results['meta']            = meta
    results['num_train']       = num_train
    results['Q_fom']           = Q_fom
    results['Q_surrogate']     = Q_surrogate
    results['Q_corrected']     = Q_corrected
    results['scores']          = scores
    results['flags']           = flags
    results['thresholds']      = monitor_param['thresholds']
    results['error']           = error
    results['error_corrected'] = error_corrected
    results['onset']           = onset
    results['detector_scores'] = detector_scores

    save_results(config, working_dir, results)
    print_summary(config, results)

    return results


def save_results(config, working_dir, results):

    dir_results = os.path.join(working_dir, f"{config['surrogate']['method']}_results")
    os.makedirs(dir_results, exist_ok=True)

    np.save(os.path.join(dir_results, 'surrogate_cons.npy'), results['Q_surrogate'])
    np.save(os.path.join(dir_results, 'corrected_cons.npy'), results['Q_corrected'])
    np.save(os.path.join(dir_results, 'error.npy'), results['error'])
    np.save(os.path.join(dir_results, 'error_corrected.npy'), results['error_corrected'])
    np.savez(os.path.join(dir_results, 'detector_scores.npz'), **results['scores'])
    np.savez(os.path.join(dir_results, 'detector_flags.npz'), **results['flags'])

    summary = {
        'case'            : config['case']['name'],
        'surrogate'       : config['surrogate'],
        'scoring'         : config['scoring'],
        'correct'         : config['correct'],
        'final_error'           : float(results['error'][-1]),
        'final_error_corrected' : float(results['error_corrected'][-1]),
        'num_train'       : results['num_train'],
        'onset_step'      : results['onset'],
        'onset_time'      : None if results['onset'] is None else float(results['meta']['time'][results['onset']]),
        'thresholds'      : {name: float(value) for name, value in results['thresholds'].items()},
        'detector_scores' : results['detector_scores'],
    }

    with open(os.path.join(dir_results, 'summary.json'), 'w') as summary_file:
        json.dump(summary, summary_file, indent=4)

    visual_func.plot_benchmark(config, results, os.path.join(dir_results, 'summary.png'))
    visual_func.plot_last_snapshot(config, results, os.path.join(dir_results, 'last_snapshot.png'))

    results['dir_results'] = dir_results


def print_summary(config, results):

    time  = results['meta']['time']
    onset = results['onset']

    print(f"Case: {config['case']['name']}   Surrogate: {config['surrogate']['method']}")
    print(f"Trained on {results['num_train']} of {time.size} snapshots (up to t = {time[results['num_train'] - 1]:.4g})")

    if onset is None:
        print('True failure onset: the surrogate never exceeded the error threshold')
    else:
        print(f"True failure onset: step {onset} (t = {time[onset]:.4g})")

    print(f"{'detector':<20}{'first flag':>12}{'delay':>8}{'false alarms':>14}   result")

    for name, score in results['detector_scores'].items():

        first_flag = '-' if score['first_flag'] is None else str(score['first_flag'])
        delay      = '-' if score['delay'] is None else str(score['delay'])
        result     = 'detected' if score['detected'] else ('missed' if score['missed'] else 'no failure')

        print(f"{name:<20}{first_flag:>12}{delay:>8}{score['false_alarms']:>14}   {result}")

    if config['correct']['method'] != 'none':

        failed = slice(onset, None)

        print(f"Correction ({config['correct']['method']}): mean error after the onset "
              f"{np.mean(results['error'][failed]):.3g} -> {np.mean(results['error_corrected'][failed]):.3g}, "
              f"largest {np.max(results['error'][failed]):.3g} -> {np.max(results['error_corrected'][failed]):.3g}")

    print(f"Results saved in {results['dir_results']}")
