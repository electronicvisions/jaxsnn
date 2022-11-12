from functools import partial
from typing import Callable, List, Tuple

import jax
import jax.numpy as np
import matplotlib.pyplot as plt

from jaxsnn.event.functional import f, forward_integration, step, transition
from jaxsnn.event.root import ttfs_solver
from jaxsnn.types import Spike

tau_mem = 1e-2
tau_syn = 5e-3
tau_mem_inv = 1 / tau_mem
tau_syn_inv = 1 / tau_syn

v_th = 0.6
t_late = tau_syn + tau_mem
t_max = 2 * t_late
A = np.array([[-tau_mem_inv, tau_mem_inv], [0, -tau_syn_inv]])


@jax.jit
def loss_fn(first_spikes, target):
    loss_value = -np.sum(np.log(1 + np.exp(-np.abs(first_spikes - target) / tau_mem)))
    return loss_value


def init_weights(rng: jax.random.KeyArray, layers: Tuple[int, int]):
    n_input, n_hidden = layers
    scale_factor = 2.0
    input_rng, hidden_rng = jax.random.split(rng)
    input_weights = jax.random.uniform(input_rng, (n_input, n_hidden)) * scale_factor
    recurrent_weights = (
        jax.random.uniform(hidden_rng, (n_hidden, n_hidden))
        * (1 - np.eye(n_hidden))
        * scale_factor
    )

    return [input_weights, recurrent_weights]


single_dynamics = partial(f, A)
dynamics = jax.vmap(single_dynamics, in_axes=(0, None))
solver = partial(ttfs_solver, tau_mem, v_th)
batched_solver = jax.vmap(solver, in_axes=(0, None))
step_fn = partial(step, dynamics, batched_solver, transition, t_max)
forward = partial(forward_integration, step_fn, 10)

# minimal forward example
def forward1(weights, input_spikes):
    t = 0
    input_weights, recurrent_weights = weights
    recording = np.array([0.0, 0.0])

    y = np.zeros((recurrent_weights.shape[0], 2))

    # first input spike
    spike_idx, t_dyn = 0, input_spikes[0]
    y = dynamics(y, t_dyn)
    y = y.at[:, 1].set(y[:, 1] + input_weights[spike_idx])
    t += t_dyn

    # second input spike
    spike_idx, t_dyn = 1, input_spikes[1] - t
    y = dynamics(y, t_dyn)
    y = y.at[:, 1].set(y[:, 1] + input_weights[spike_idx])
    t += t_dyn

    # first internal spike
    spike_times = batched_solver(y, 1e-3)
    spike_idx = np.nanargmin(spike_times)
    t_dyn = spike_times[spike_idx]
    t += t_dyn
    recording = recording.at[0].set(t)  # type: ignore
    y = dynamics(y, t_dyn)

    y = y.at[spike_idx, 0].set(0.0)
    y = y.at[:, 1].set(y[:, 1] + recurrent_weights[spike_idx])

    # second internal spike
    spike_times = batched_solver(y, 1e-4)
    spike_idx = np.nanargmin(spike_times)
    t_dyn = spike_times[spike_idx]
    t += t_dyn
    recording = recording.at[1].set(t)
    return recording[0] + recording[1]


def forward3(weights, input_spikes):
    state, spikes = forward(weights, input_spikes)
    return spikes.time[2] + spikes.time[3]


def forward4(weights, input_spikes):
    state, spikes = forward(weights, input_spikes)
    time1 = np.nanmin(np.where(spikes.idx == 3, spikes.time, np.nan))
    time2 = np.nanmin(np.where(spikes.idx == 1, spikes.time, np.nan))
    return time1 + time2


def forward5(weights, batch):
    input_spikes, target = batch
    state, spikes = forward(weights, input_spikes)

    time1 = np.nanmin(np.where(spikes.idx == 3, spikes.time, np.nan))
    time2 = np.nanmin(np.where(spikes.idx == 1, spikes.time, np.nan))

    return (
        loss_fn(np.array([time1, time2]), target),
        (np.array([time1, time2]), (spikes.time, spikes.idx)),
    )


def inspect(rng, n_input, n_hidden, forward, input_spikes):
    weights = init_weights(rng, (n_input, n_hidden))
    print(forward(weights, input_spikes))
    grad = jax.grad(forward)(weights, input_spikes)
    print(grad)


def assert_vals_equal(rng, n_input, n_hidden, funcs: List[Callable], input_spikes):
    weights = init_weights(rng, (n_input, n_hidden))
    for fn in funcs:
        t_spikes = fn(weights, input_spikes)
        assert t_spikes == 0.003840894


def update(weights, batch):
    value, grad = jax.value_and_grad(forward5, has_aux=True)(weights, batch)
    # if any([(np.isnan(g)) for g in grad]):
    # assert False
    weights = jax.tree_map(lambda f, df: f - 0.1 * df, weights, grad)
    return weights, value


def plot_spike_times(spikes, axs):
    for i in range(4):
        it = i * 200
        spike_times = spikes[0][it]
        spike_idx = spikes[1][it]
        axs[i].scatter(
            x=spike_times,
            y=spike_idx + 1,
            s=3 * (120.0 / len(spike_times)) ** 2.0,
            marker="|",
            c="black",
        )
        axs[i].set_ylabel("neuron id")
        axs[i].set_xlabel(r"$t$ [us]")


def train(trainset):
    n_input = 2
    n_hidden = 4
    rng = jax.random.PRNGKey(42)
    weights = init_weights(rng, (n_input, n_hidden))

    # state, (times, spikes) = forward5(
    #     weights, (Spike(trainset[0].time[0], trainset[0].idx[0]), trainset[1][0])
    # )
    # print(spikes)
    weights, (loss, (t_output, spikes)) = jax.lax.scan(update, weights, trainset)

    fix, (ax1, ax2) = plt.subplots(2, 4, figsize=(18, 6))
    ax1[0].plot(np.arange(len(loss)), loss, label="Loss")
    ax1[1].plot(np.arange(len(t_output)), t_output[:, 0] / t_max, label="t_spike 1")
    ax1[1].plot(np.arange(len(t_output)), t_output[:, 1] / t_max, label="t_spike 2")
    ax1[1].legend()
    plot_spike_times(spikes, ax2)
    plt.show()


def constant_dataset(n_epochs):
    input_spikes = Spike(
        np.array([0.1, 0.2, 1]) * t_max,  # type: ignore
        np.array([0, 1, 0]),
    )
    target = np.array([0.2, 0.4]) * t_max  # type: ignore
    batch = (input_spikes, target)
    tiling = (n_epochs, 1)
    dataset = (
        Spike(np.tile(batch[0].time, tiling), np.tile(batch[0].idx, tiling)),
        np.tile(batch[1], tiling),
    )
    return dataset


if __name__ == "__main__":
    n_epochs = 1000
    trainset = constant_dataset(1000)
    train(trainset)
