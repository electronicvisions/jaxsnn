from functools import partial

import jax
import jax.numpy as np
import matplotlib.pyplot as plt
from jaxsnn.event.functional import (
    f,
    forward_integration,
    jump_condition,
    step,
    tr_equation,
)
from jaxsnn.event.root import (
    batched_ttfs_solver,
    cr_newton_solver,
    newton_solver,
    ttfs_solver,
)

tau_mem = 1e-3
tau_syn = 5e-4
tau_mem_inv = 1 / tau_mem
tau_syn_inv = 1 / tau_syn
v_th = 0.3

A = np.array([[-tau_mem_inv, tau_mem_inv], [0, -tau_syn_inv]])


def loss_fn(first_spikes, target):
    loss_value = -np.sum(np.log(1 + np.exp(-np.abs(first_spikes - target) / tau_mem)))
    return loss_value


def init_weights(rng, n_neurons):
    off_diagonal = 1 - np.eye(n_neurons)
    weights = jax.random.uniform(rng, (n_neurons, n_neurons)) * 2.0
    # return off_diagonal * weights
    return np.array(
        [
            [0.0, 0.5, 1.0, 0.8],
            [0.4, 0.0, 1.1, 0.9],
            [0.4, 0.1, 0.0, 0.6],
            [0.4, 0.1, 0.2, 0.0],
        ]
    )


def inspect():
    n_neurons = 4

    single_dynamics = partial(f, A)
    dynamics = jax.vmap(single_dynamics, in_axes=(0, None))
    solver = partial(ttfs_solver, tau_mem, v_th)
    batched_solver = jax.vmap(solver, in_axes=(0, None))

    # minimual forward examples
    def forward1(weights):
        recording = np.array([0.0, 0.0])

        y0 = np.zeros((n_neurons, 2))

        # first round
        spike_idx, t_dyn = 0, 1e-4
        y_minus = dynamics(y0, t_dyn)
        y0 = tr_equation(weights, y_minus, spike_idx)

        # second round
        spike_idx, t_dyn = 1, 2e-5
        y_minus = dynamics(y0, t_dyn)
        y0 = tr_equation(weights, y_minus, spike_idx)
        t = 1.2e-4

        # first internal apike
        spike_times = batched_solver(y0, 1e-4)
        spike_idx = np.nanargmin(spike_times)
        t_dyn = spike_times[spike_idx]
        t += t_dyn
        recording = recording.at[0].set(t)  # type: ignore
        y_minus = dynamics(y0, t_dyn)
        y0 = tr_equation(weights, y_minus, spike_idx)

        # second internal spike
        spike_times = batched_solver(y0, 1e-4)
        spike_idx = np.nanargmin(spike_times)
        t_dyn = spike_times[spike_idx]
        t += t_dyn
        recording = recording.at[1].set(t)
        return recording[0] + recording[1]

    # rng = jax.random.PRNGKey(42)
    # weights = init_weights(rng, n_neurons)
    # print(forward1(weights))
    # grad = jax.grad(forward1)(weights)
    # print(grad)

    # now user our step fn
    step_fn = partial(step, dynamics, batched_solver, tr_equation)

    def forward2(weights):
        input_spikes = np.array([1e-4, 1.2e-4, np.inf, np.inf])
        y0 = np.zeros((n_neurons, 2))
        t = 0
        times = np.zeros(4)
        for i in range(4):
            y0, t_dyn, spike_idx = step_fn(weights, input_spikes - t, y0, 0.1)  # type: ignore
            t += t_dyn
            times = times.at[i].set(t)
        return times[2] + times[3]

    # weights = init_weights(rng, n_neurons)
    # print(forward2(weights))
    # grad = jax.grad(forward2)(weights)
    # print(grad)

    # now use forward integration
    def forward3(weights):
        t_max = 0.1
        input_spikes = np.array([1e-4, 1.2e-4, np.inf, np.inf])
        forward_int = partial(forward_integration, step_fn, 4)
        state, (spike_times, spike_idx) = forward_int(weights, input_spikes, t_max)
        return spike_times[2] + spike_times[3]

    # weights = init_weights(rng, n_neurons)
    # print(forward3(weights))
    # grad = jax.grad(forward3)(weights)
    # print(grad)

    # now determine first spike
    def forward4(weights, target):
        t_max = 0.1
        input_spikes = np.array([1e-4, 1.2e-4, np.inf, np.inf])
        forward_int = partial(forward_integration, step_fn, 4)
        state, (spike_times, spike_idx) = forward_int(weights, input_spikes, t_max)
        time1 = np.nanmin(np.where(spike_idx == 2, spike_times, np.nan))
        time2 = np.nanmin(np.where(spike_idx == 3, spike_times, np.nan))
        return loss_fn(np.array([time1, time2]), target), np.array([time1, time2])

    forward4 = jax.jit(forward4)

    # weights = init_weights(rng, n_neurons)
    # print(forward4(weights))
    # grad = jax.grad(forward4)(weights)
    # print(grad)

    def train():
        rng = jax.random.PRNGKey(42)
        target = np.array([tau_syn, 1.4 * tau_syn])
        weights = init_weights(rng, n_neurons)
        for i in range(100):
            value, grad = jax.value_and_grad(forward4, has_aux=True)(weights, target)
            weights = jax.tree_map(lambda f, df: f - 0.1 * df, weights, grad)
            print(forward4(weights, target)[1], target)

    train()


if __name__ == "__main__":
    inspect()
