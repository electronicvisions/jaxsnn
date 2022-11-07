from functools import partial
from typing import Callable, List, Tuple

import jax
import jax.numpy as np

from jaxsnn.event.functional import f, forward_integration, step
from jaxsnn.event.root import ttfs_solver

tau_mem = 1e-2
tau_syn = 5e-3
tau_mem_inv = 1 / tau_mem
tau_syn_inv = 1 / tau_syn

v_th = 0.3
t_late = tau_syn + tau_mem
t_max = 2 * t_late
A = np.array([[-tau_mem_inv, tau_mem_inv], [0, -tau_syn_inv]])


def loss_fn(first_spikes, target):
    loss_value = -np.sum(np.log(1 + np.exp(-np.abs(first_spikes - target) / tau_mem)))
    return loss_value


def init_weights(rng: jax.random.KeyArray, layers: Tuple[int, int]):
    n_input, n_hidden = layers
    scale_factor = 2.0
    input_rng, hidden_rng = jax.random.split(rng)
    input_weights = jax.random.normal(input_rng, (n_input, n_hidden)) * scale_factor
    recurrent_weights = (
        jax.random.normal(hidden_rng, (n_hidden, n_hidden))
        * (1 - np.eye(n_hidden))
        * scale_factor
    )

    return [input_weights, recurrent_weights]


single_dynamics = partial(f, A)
dynamics = jax.vmap(single_dynamics, in_axes=(0, None))
solver = partial(ttfs_solver, tau_mem, v_th)
batched_solver = jax.vmap(solver, in_axes=(0, None))
step_fn = partial(step, dynamics, batched_solver)

rng = jax.random.PRNGKey(42)
n_input = 2
n_hidden = 4


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


def forward2(weights, input_spikes):
    y0 = np.zeros((weights[1].shape[0], 2))
    t = 0
    times = np.zeros(4)
    for i in range(4):
        y0, t_dyn, spike_idx = step_fn(weights, input_spikes - t, y0, t_max)  # type: ignore
        t += t_dyn
        times = times.at[i].set(t)
    return times[2] + times[3]


def forward3(weights, input_spikes):
    forward_int = partial(forward_integration, step_fn, 4)
    state, (spike_times, spike_idx) = forward_int(weights, input_spikes, t_max)
    return spike_times[2] + spike_times[3]


def forward4(weights, input_spikes):
    forward_int = partial(forward_integration, step_fn, 4)
    state, (spike_times, spike_idx) = forward_int(weights, input_spikes, t_max)
    time1 = np.nanmin(np.where(spike_idx == 3, spike_times, np.nan))
    time2 = np.nanmin(np.where(spike_idx == 1, spike_times, np.nan))
    return time1 + time2


def forward5(weights, batch):
    input_spikes, target = batch
    forward_int = partial(forward_integration, step_fn, 4)
    state, (spike_times, spike_idx) = forward_int(weights, input_spikes, t_max)
    time1 = np.nanmin(np.where(spike_idx == 3, spike_times, np.nan))
    time2 = np.nanmin(np.where(spike_idx == 1, spike_times, np.nan))
    return loss_fn(np.array([time1, time2]), target), np.array([time1, time2])


def inspect(forward, input_spikes):
    weights = init_weights(rng, (n_input, n_hidden))
    print(forward(weights, input_spikes))
    grad = jax.grad(forward)(weights, input_spikes)
    print(grad)


def assert_vals_equal(funcs: List[Callable], input_spikes):
    weights = init_weights(rng, (n_input, n_hidden))
    for fn in funcs:
        t_spikes = fn(weights, input_spikes)
        assert t_spikes == 0.003840894


def train(input_spikes):
    rng = jax.random.PRNGKey(42)
    target = np.array([0.1, 0.3]) * t_max  # type: ignore
    batch = (input_spikes, target)
    n_input = 2
    n_hidden = 4
    weights = init_weights(rng, (n_input, n_hidden))
    for i in range(100):
        value, grad = jax.value_and_grad(forward5, has_aux=True)(weights, batch)
        weights = jax.tree_map(lambda f, df: f - 0.1 * df, weights, grad)
        print(forward5(weights, batch)[1], target)


if __name__ == "__main__":
    input_spikes = np.array([0.01, 0.04, np.inf, np.inf]) * t_max  # type: ignore
    assert_vals_equal([forward1, forward2, forward3, forward4], input_spikes)
    train(input_spikes)
