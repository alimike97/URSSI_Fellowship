import copy
import tomllib

# default value of every optional keyword, section by section
DEFAULTS = {

    'case': {
        'name'            : 'case',
        'start_iter'      : 0,
        'step_iter'       : 1,
        'end_iter'        : None,
        'save_interval'   : 1,
        'cons_volume_weighted' : False,
        'dt'              : 1.0,
        'x_initial'       : 0.0,
        'x_final'         : 1.0,
        'gamma'           : 1.4,
        'cons_var_names'  : ['Density', 'Momentum', 'Total Energy'],
    },

    'surrogate': {
        'method'          : 'opinf',
        'train_end_iter'  : None,
        'num_modes'       : 10,
        'regularization'  : 1e-2,
        'operators'       : 'cAH',
        'regressor'       : 'gpr',
        'hidden_channels' : 16,
        'epochs'          : 400,
        'learning_rate'   : 3e-3,
        'train_noise'     : 1e-3,
        'seed'            : 0,
    },

    'detect': {
        'detectors'       : ['finite', 'positivity', 'conservation', 'total_variation', 'residual', 'mahalanobis'],
        'num_modes'       : 10,
        'calibrate'       : True,
        'calibration_factor' : 1.5,
        'thresholds'      : {},
    },

    'scoring': {
        'error_norm'      : 'l1',
        'error_variable'  : 'worst',
        'error_threshold' : 0.05,
    },

    'observe': {
        'obs_indices'     : 'all',
        'obs_step'        : 1,
        'obs_std'         : 1e-3,
    },

    'correct': {
        'method'          : 'none',
        'num_neighbors'   : 5,
        'library'         : 'all',
        'library_step'    : 5,
    },
}

# keywords with no sensible default
REQUIRED = {
    'case': ['fom_dir'],
}

ALLOWED = {
    'surrogate.method'       : ['opinf', 'pod_regression', 'cnn'],
    'surrogate.regressor'    : ['gpr', 'kernel_ridge'],
    'scoring.error_norm'     : ['l1', 'l2', 'linf'],
    'correct.method'         : ['none', 'projection', 'memory'],
    'correct.library'        : ['train', 'all'],
}


def read_input_file(path):

    with open(path, 'rb') as input_file:
        input_param = tomllib.load(input_file)

    return input_param


def init_config(input_param):

    config = copy.deepcopy(DEFAULTS)

    for section, values in input_param.items():

        if section not in DEFAULTS:
            raise KeyError(f"Unknown section [{section}] in input file.")

        for keyword, value in values.items():

            if keyword not in DEFAULTS[section] and keyword not in REQUIRED.get(section, []):
                raise KeyError(f"Unknown keyword '{keyword}' in section [{section}].")

            config[section][keyword] = value

    for section, keywords in REQUIRED.items():
        for keyword in keywords:
            if keyword not in config[section]:
                raise KeyError(f"Missing required keyword '{keyword}' in section [{section}].")

    for key, options in ALLOWED.items():
        section, keyword = key.split('.')
        if config[section][keyword] not in options:
            raise ValueError(f"'{config[section][keyword]}' is not a valid {key}. Options: {options}")

    # 'worst' takes the largest error over all variables, otherwise a variable name is expected
    error_variable = config['scoring']['error_variable']
    if error_variable != 'worst' and error_variable not in config['case']['cons_var_names']:
        raise ValueError(f"scoring.error_variable '{error_variable}' is not 'worst' or one of cons_var_names.")

    return config
