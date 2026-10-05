import numpy as np
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

SURFACE   = '#fcfcfb'
INK       = '#0b0b0b'
INK_MUTED = '#898781'
GRID      = '#e4e3df'
BLUE      = '#2a78d6'
ORANGE    = '#eb6834'
AQUA      = '#1baf7a'


def style_axis(ax):

    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, labelsize=9)

    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)

    for side in ['left', 'bottom']:
        ax.spines[side].set_color(GRID)


def mark_events(ax, time, num_train, onset):

    # end of the training window and true failure onset
    ax.axvline(time[num_train - 1], color=INK_MUTED, linewidth=1, linestyle=':')

    if onset is not None:
        ax.axvline(time[onset], color=INK, linewidth=1, linestyle='--')


def plot_benchmark(config, results, path):

    time      = results['meta']['time']
    num_train = results['num_train']
    onset     = results['onset']
    flags     = results['flags']
    names     = list(flags.keys())
    threshold = config['scoring']['error_threshold']

    fig, axs = plt.subplots(2, 1, figsize=(9, 7), height_ratios=[1.0, 0.25 + 0.22 * len(names)])
    fig.set_facecolor(SURFACE)

    ### error against the FOM ###
    ax = axs[0]
    style_axis(ax)

    error = np.where(np.isfinite(results['error']), results['error'], np.nan)

    corrected = config['correct']['method'] != 'none'

    ax.plot(time, error, color=ORANGE, linewidth=2, label='Surrogate')

    if corrected:
        ax.plot(time, results['error_corrected'], color=AQUA, linewidth=2, label='Corrected')
        ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc='center left')
    ax.axhline(threshold, color=INK_MUTED, linewidth=1)
    ax.text(time[0], threshold, f" failure threshold ({threshold:g})", color=INK_MUTED, fontsize=9, va='bottom')
    mark_events(ax, time, num_train, onset)

    top = ax.get_ylim()[1]
    ax.text(time[num_train - 1], top, 'end of training ', color=INK_MUTED, fontsize=9, ha='right', va='top')

    if onset is not None:
        ax.text(time[onset], top, ' true failure onset', color=INK, fontsize=9, ha='left', va='top')

    ax.set_ylabel(f"Relative {config['scoring']['error_norm'].upper()} error", color=INK, fontsize=10)
    ax.set_title(f"{config['case']['name']} - {config['surrogate']['method']} surrogate: error against the FOM",
                 color=INK, fontsize=11, loc='left')

    ### when each detector is flagging ###
    ax = axs[1]
    style_axis(ax)
    ax.grid(False, axis='y')

    step = np.diff(time, append=2 * time[-1] - time[-2])

    for row, name in enumerate(names):
        flagged = np.flatnonzero(flags[name])
        ax.barh(np.full(flagged.size, row), step[flagged], left=time[flagged], height=0.5, color=INK_MUTED, linewidth=0)

    mark_events(ax, time, num_train, onset)

    ax.set_yticks(range(len(names)), names)
    ax.tick_params(axis='y', colors=INK, labelsize=9)
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.set_xlim(axs[0].get_xlim())
    ax.set_xlabel('Time', color=INK, fontsize=10)
    ax.set_title('When each detector is flagging', color=INK, fontsize=11, loc='left')

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_last_snapshot(config, results, path):

    time      = results['meta']['time']
    x         = results['meta']['x']
    var_names = config['case']['cons_var_names']
    corrected = config['correct']['method'] != 'none'

    # last snapshot the surrogate could produce
    finite_steps = np.flatnonzero(np.all(np.isfinite(results['Q_surrogate']), axis=(1, 2)))
    last         = finite_steps[-1] if finite_steps.size > 0 else 0

    # one panel for each variable
    fig, axs = plt.subplots(len(var_names), 1, figsize=(9, 3 * len(var_names)), sharex=True, squeeze=False)
    fig.set_facecolor(SURFACE)

    for var, ax in enumerate(axs[:, 0]):

        style_axis(ax)

        ax.plot(x, results['Q_fom'][last, var], color=INK, linewidth=2,ls='-' ,label='FOM')
        ax.plot(x, results['Q_surrogate'][last, var], color=ORANGE, linewidth=2, ls='--' ,label='Surrogate')

        if corrected:
            ax.plot(x, results['Q_corrected'][last, var], color=AQUA, linewidth=2, ls='--' ,label='Corrected')

        ax.set_ylabel(var_names[var], color=INK, fontsize=10)

    axs[0, 0].set_title(f"{config['case']['name']} - {config['surrogate']['method']} surrogate: last snapshot, t = {time[last]:.4g}",
                        color=INK, fontsize=11, loc='left')
    axs[0, 0].legend(frameon=False, fontsize=9, labelcolor=INK)
    axs[-1, 0].set_xlabel('x', color=INK, fontsize=10)

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
