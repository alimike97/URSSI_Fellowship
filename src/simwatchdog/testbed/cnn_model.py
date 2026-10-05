import numpy as np
import torch

# Neural network surrogate: a small 1D CNN that predicts the change of the state over one saved step
# and is applied repeatedly from the initial condition.


def init_network(num_var, hidden_channels):

    def conv(channels_in, channels_out):
        return torch.nn.Conv1d(channels_in, channels_out, kernel_size=5, padding=2, padding_mode='replicate')

    network = torch.nn.Sequential(
        conv(num_var, hidden_channels), torch.nn.GELU(),
        conv(hidden_channels, hidden_channels), torch.nn.GELU(),
        conv(hidden_channels, hidden_channels), torch.nn.GELU(),
        conv(hidden_channels, num_var),
    )

    return network.double()


def train(config, Q_train, time_train):

    surrogate = config['surrogate']

    torch.manual_seed(surrogate['seed'])

    surrogate_param = {}

    # states are scaled to order one, the change over one step is scaled separately since it is much smaller
    var_scale   = np.max(np.abs(Q_train), axis=(0, 2))[:, None]
    delta_scale = np.std(np.diff(Q_train, axis=0) / var_scale)

    state  = torch.from_numpy(Q_train[:-1] / var_scale)
    target = torch.from_numpy(np.diff(Q_train, axis=0) / var_scale / delta_scale)

    network   = init_network(Q_train.shape[1], surrogate['hidden_channels'])
    optimizer = torch.optim.Adam(network.parameters(), lr=surrogate['learning_rate'])

    for epoch in range(surrogate['epochs']):

        optimizer.zero_grad()

        # noise on the input makes the rollout less sensitive to its own errors
        noisy_state = state + surrogate['train_noise'] * torch.randn_like(state)

        loss = torch.mean((network(noisy_state) - target) ** 2)

        loss.backward()
        optimizer.step()

    print(f"CNN training loss after {surrogate['epochs']} epochs: {loss.item():.3e}")

    surrogate_param['network']     = network
    surrogate_param['var_scale']   = var_scale
    surrogate_param['delta_scale'] = delta_scale

    return surrogate_param


def rollout(surrogate_param, Q_initial, time):

    network     = surrogate_param['network']
    var_scale   = surrogate_param['var_scale']
    delta_scale = surrogate_param['delta_scale']

    Q_surrogate = np.full((time.size, *Q_initial.shape), np.nan)

    state = torch.from_numpy(Q_initial / var_scale)[None]

    with torch.no_grad():

        for step in range(time.size):

            Q_surrogate[step] = state[0].numpy() * var_scale

            state = state + delta_scale * network(state)

    return Q_surrogate
