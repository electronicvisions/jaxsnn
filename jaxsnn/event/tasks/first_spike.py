import time
from functools import partial
from typing import Callable, List, Tuple

import jax
import jax.numpy as np
import matplotlib.pyplot as plt

from jaxsnn.event.functional import f, forward_integration, step
from jaxsnn.event.leaky_integrate import leaky_integrator, nll_loss
from jaxsnn.event.root import ttfs_solver
from jaxsnn.types import Array

tau_mem = 1e-2
tau_syn = 5e-3
tau_mem_inv = 1 / tau_mem
tau_syn_inv = 1 / tau_syn
v_th = 0.3
t_late = tau_syn + tau_mem
t_max = 2 * t_late

A = np.array([[-tau_mem_inv, tau_mem_inv], [0, -tau_syn_inv]])


def ttfs_loss(
    forward: Callable,
    weights: List[Array],
    batch: Tuple[Array, Array],
):
    input_spikes, targets = batch

    (t, y_new), (spike_times, spike_idx) = forward(weights[:2], input_spikes, t_max)

    first_spike1 = np.nanmin(np.where(spike_idx == 0, spike_times, np.nan))
    first_spike2 = np.nanmin(np.where(spike_idx == 1, spike_times, np.nan))

    first_spikes = np.array([first_spike1, first_spike2])
    loss = np.sum(((targets - first_spikes) / t_max) ** 2)
    loss = -np.sum(np.log(1 + np.exp(-np.abs(first_spikes - targets) / t_max)))
    return loss, np.argmin(targets) == np.argmin(first_spikes)


def init_weights(rng: jax.random.KeyArray, n_input: int, n_hidden: int):
    scale_factor = 2.0
    input_rng, hidden_rng = jax.random.split(rng)
    input_weights = jax.random.normal(input_rng, (n_input, n_hidden)) * scale_factor
    recurrent_weights = (
        jax.random.normal(hidden_rng, (n_hidden, n_hidden))
        * (1 - np.eye(n_hidden))
        * scale_factor
    )

    return [input_weights, recurrent_weights]


def linear_dataset(rng: jax.random.KeyArray, iterations: int):
    input = jax.random.uniform(rng, (iterations, 2))
    which_class = (input[:, 0] < input[:, 1]).astype(int)
    encoding = np.array([[0, 1], [1, 0]]) * t_max  # type: ignore

    # augment dataset
    input = np.hstack((input, 1 - input))
    input = input * t_late  # type: ignore
    return (input, encoding[which_class])


def circle_dataset(rng: jax.random.KeyArray, iterations: int):
    input = jax.random.uniform(rng, (iterations, 2))
    radius = np.sqrt(0.5 / np.pi)  # spread classes equal
    center = (0.5, 0.5)
    which_class = (
        (input[:, 0] - center[0]) ** 2 + (input[:, 1] - center[1]) ** 2 <= radius**2
    ).astype(int)
    encoding = np.array([[0, 1], [1, 0]]) * t_max  # type: ignore

    input = np.hstack((input, 1 - input))  # augment dataset
    input = input * t_late  # type: ignore
    return (input, encoding[which_class])


def max_over_time_loss_single(weights: List[Array], spikes: Array, targets: Array):
    ts = np.arange(0, 1e-2, 1e-4)
    output = leaky_integrator(A, weights, spikes, ts)
    max_voltage = np.max(output[::, 0], axis=0)
    loss_value = nll_loss(max_voltage, targets)
    acc_value = np.sum(np.argmax(max_voltage) == np.argmax(targets))
    return loss_value, acc_value


def regularization_loss(spikes, n_hidden, expected_spikes, rho=1e-4):
    (spike_times, spike_idx) = spikes
    spikes_per_neuron = np.sum(np.eye(n_hidden)[spike_idx], axis=1)
    mse_loss = np.sum(np.square(spikes_per_neuron - expected_spikes))
    return rho * mse_loss


def update(
    loss_fn: Callable,
    weights: List[Array],
    batch: Tuple[Array, Array],
):
    value, grad = jax.value_and_grad(loss_fn, has_aux=True)(weights, batch)
    weights = jax.tree_map(lambda w, g: w - 1.0 * g, weights, grad)
    return weights, value


def train():
    n_spikes = 20
    n_input = 4
    n_hidden = 6

    single_dynamics = partial(f, A)
    dynamics = jax.vmap(single_dynamics, in_axes=(0, None))

    solver = partial(ttfs_solver, tau_mem, v_th)
    solver = jax.vmap(solver, in_axes=(0, None))

    # define forward function
    step_fn = partial(step, dynamics, solver)
    forward = jax.jit(partial(forward_integration, step_fn, n_spikes))
    loss_fn = partial(ttfs_loss, forward)
    update_fn = partial(update, loss_fn)

    # init params
    rng = jax.random.PRNGKey(42)
    weights = init_weights(rng, n_input, n_hidden)
    rng, trainset_rng, testset_rng = jax.random.split(rng, 3)
    testset = linear_dataset(testset_rng, 1_000)
    trainset = linear_dataset(trainset_rng, 1_000)

    loss_fn(weights, (testset[0][0], testset[1][0]))
    test_loss, test_acc = jax.vmap(loss_fn, in_axes=(None, 0))(weights, testset)
    print(f"Loss: {np.mean(test_loss)}, acc: {np.mean(test_acc)}")

    # manually for debugging
    loss_value = []
    for i in range(len(trainset[0])):
        weights, (loss, acc) = update_fn(weights, (trainset[0][i], trainset[1][i]))
        loss_value.append(loss)

    # or with scan
    # weights, (loss_value, acc) = jax.lax.scan(update, weights, trainset)

    test_loss, test_acc = jax.vmap(loss_fn, in_axes=(None, 0))(weights, testset)
    print(f"Loss: {np.mean(test_loss)}, acc: {np.mean(test_acc)}")

    plt.plot(np.arange(len(loss_value)), loss_value)
    plt.show()


if __name__ == "__main__":
    start = time.time()
    train()
    print(f"Training took {time.time() - start:.3f} seconds")
