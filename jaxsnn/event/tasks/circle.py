import time
from functools import partial
from turtle import circle
from typing import Tuple

import jax
import jax.numpy as np
import matplotlib.pyplot as plt
from jax.tree_util import tree_flatten
from jaxsnn.event.functional import (
    f,
    forward_integration,
    jump_condition,
    step,
    tr_equation,
)
from jaxsnn.event.leaky_integrate import leaky_integrator, nll_loss
from jaxsnn.event.root import cr_newton_solver, newton_solver, ttfs_solver

tau_mem = 1e-3
tau_syn = 5e-4
tau_mem_inv = 1 / tau_mem
tau_syn_inv = 1 / tau_syn
v_th = 0.3

A = np.array([[-tau_mem_inv, tau_mem_inv], [0, -tau_syn_inv]])


def init_weights(rng, n_neurons):
    off_diagonal = 1 - np.eye(n_neurons)
    weights = jax.random.uniform(rng, (n_neurons, n_neurons)) * 2.0
    return off_diagonal * weights


def init_net(rng, layer_sizes):
    rng, layer_rng = jax.random.split(rng)
    return (
        init_weights(rng, layer_sizes[0]),
        jax.random.uniform(layer_rng, (layer_sizes[0], layer_sizes[1])) * 2.0,
    )


def first_spike(spikes: Tuple, idx: int):
    """Return spike times"""
    spike_times, spike_idx = spikes
    output_idx = np.where(spike_idx == idx)[0]
    return spike_times[output_idx][0]


def max_over_time_loss(weights, spikes, targets):
    ts = np.arange(0, 1e-2, 1e-4)
    output = leaky_integrator(A, weights[1], spikes, ts)
    max_voltage = np.max(output[::, 0], axis=0)
    loss_value = nll_loss(max_voltage, targets)
    return loss_value


def loss_and_spikes(forward, weights, batch):
    input_spikes, targets = batch
    t_max = 100 * tau_mem

    (t, y), spikes = forward(weights[0], input_spikes, t_max)

    # ttfs loss
    # first_spike1 = np.nanmin(np.where(spike_idx == 2, spike_times, np.nan), axis=1)
    # first_spike2 = np.nanmin(np.where(spike_idx == 3, spike_times, np.nan), axis=1)
    # first_spikes = np.stack((first_spike1, first_spike2), axis=1)
    # loss_value = -np.sum(np.log(1 + np.exp(-np.abs(first_spikes - target) / tau_mem)))

    batched_loss = jax.vmap(max_over_time_loss, in_axes=(None, ((0, 0)), 0))
    loss_value = batched_loss(weights, spikes, targets)

    return np.mean(loss_value) * 1000, spikes


def simple_dataset(samples, batch_size):
    input_spikes = np.array([1e-4, 1.2e-4, np.inf, np.inf])
    input_spikes = np.repeat(input_spikes[None, :], batch_size, axis=0)  # type: ignore

    target_times = np.array([1, 0])
    target_times = np.repeat(target_times[None, :], batch_size, axis=0)  # type: ignore

    return input_spikes, target_times


def circle_dataset(key, batch_size, n_batches):
    r = 1
    t_late = 3 * tau_syn
    x = jax.random.uniform(key, (n_batches, batch_size, 2))
    y = np.where(x[:, :, 0] ** 2 + x[:, :, 1] ** 2 > 1, 1, 0)
    one_hot = np.array([[0.0, tau_mem], [tau_mem, 0.0]])[y]  # type: ignore
    return np.dstack((x, np.full_like(x, np.inf))) * t_late, one_hot  # type: ignore


def train():
    single_dynamics = partial(f, A)
    dynamics = jax.vmap(single_dynamics, in_axes=(0, None))
    jc = partial(jump_condition, single_dynamics, v_th)

    # define solver
    solver = partial(ttfs_solver, tau_mem, v_th)
    # solver = partial(newton_solver, jc)
    # solver = partial(cr_newton_solver, jc)

    # solver for multiple neurons
    solver = jax.jit(jax.vmap(solver, in_axes=(0, None)))

    # define step function function
    step_fn = partial(step, dynamics, solver, tr_equation)

    # define forward function
    n_spikes = 20
    forward = jax.jit(partial(forward_integration, step_fn, n_spikes))
    batched_forward = jax.vmap(forward, in_axes=(None, 0, None))

    loss_fn = partial(loss_and_spikes, batched_forward)

    # training parameters
    rng = jax.random.PRNGKey(42)
    rng, dataset_rng = jax.random.split(rng)
    n_neurons = 4
    n_outputs = 2
    weights = init_net(rng, (n_neurons, n_outputs))

    # define input and target and train one epoch
    dataset = circle_dataset(dataset_rng, 128, 10)

    # define update step
    def update(weights, batch):
        (loss, spikes), grad = jax.value_and_grad(loss_fn, has_aux=True)(weights, batch)  # type: ignore
        # assert all([not np.any(np.isnan(g)) for g in grad])
        # print(loss)
        # debug
        # (spike_times, spike_idx) = spikes
        # spikes_per_neuron = [
        #     int(np.mean(np.count_nonzero(spike_idx == i, axis=1)))
        #     for i in range(n_neurons)
        # ]
        # print(
        #     f"{sum(spikes_per_neuron)} spikes {spikes_per_neuron}, avg gradient: {[np.mean(g) for g in grad]}, avg weight: {[np.mean(w) for w in weights]}"
        # )

        # update
        weights = jax.tree_map(lambda f, df: f - 0.1 * df, weights, grad)
        return weights, (loss, spikes)

    # train
    # loss_array = []
    # for i in range(len(dataset[0])):
    #     weights, (loss, spikes) = update(weights, (dataset[0][i], dataset[1][i]))
    #     loss_array.append(loss)
    weights, (loss_array, spikes) = jax.lax.scan(update, weights, dataset)

    plt.plot(np.arange(len(loss_array)), loss_array)
    plt.show()


if __name__ == "__main__":
    start = time.time()
    train()
    print(f"Training took {time.time() - start:.3f} seconds")


# TODO: speed up
# ttfs solver api
