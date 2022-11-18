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
last_layer = partial(leaky_integrator, A)


def last_voltage(weights: Array, spikes: Array, targets: Array):
    # T = 100
    # ts = np.linspace(0, t_max, T)

    output = last_layer(weights, spikes, np.array([t_max]))
    last_voltage = output[0, :, 0]
    loss_value = nll_loss(last_voltage, targets)
    acc_value = np.sum(np.argmax(last_voltage) == np.argmax(targets))
    return loss_value, acc_value, last_voltage


def max_over_time_loss(
    forward: Callable, weights: List[Array], batch: Tuple[Array, Array]
):
    input_spikes, targets = batch
    y_new, spikes = forward(weights[:2], input_spikes, t_max)
    loss, acc, max_v = last_voltage(weights[2], spikes, targets)
    return loss, (acc, spikes, max_v)


def update(loss_fn: Callable, weights: List[Array], batch: Tuple[Array, Array]):
    value, grad = jax.value_and_grad(loss_fn, has_aux=True)(weights, batch)
    lr = 0.1
    scaling = [1 / tau_mem, 1 / tau_mem, 1]
    weights = [w - lr * g * s for w, g, s, in zip(weights, grad, scaling)]
    return weights, value


def init_weights(rng: jax.random.KeyArray, layers: Tuple[int, int, int]):
    scaling = 1.0
    n_input, n_hidden, n_output = layers
    input_rng, hidden_rng, output_rng = jax.random.split(rng, 3)
    input_weights = jax.random.uniform(input_rng, (n_input, n_hidden)) * scaling
    recurrent_weights = (
        jax.random.normal(hidden_rng, (n_hidden, n_hidden))
        * (1 - np.eye(n_hidden))
        * scaling
    )
    output_weights = jax.random.normal(output_rng, (n_hidden, n_output)) * scaling

    return [input_weights, recurrent_weights, output_weights]


def linear_dataset(rng: jax.random.KeyArray, iterations: int):
    input = jax.random.uniform(rng, (iterations, 2))
    which_class = (input[:, 0] < input[:, 1]).astype(int)
    encoding = np.array([[0, 1], [1, 0]])

    # augment dataset
    input = np.hstack((input, 1 - input))
    input = input * t_late  # type: ignore
    return (input, encoding[which_class])


if __name__ == "__main__":
    n_input = 4
    n_hidden = 4
    n_output = 2
    n_spikes = 20
    batch_size = 100

    single_dynamics = partial(f, A)
    dynamics = jax.vmap(single_dynamics, in_axes=(0, None))

    solver = partial(ttfs_solver, tau_mem, v_th)
    solver = jax.vmap(solver, in_axes=(0, None))

    # define forward function
    step_fn = partial(step, dynamics, solver)
    forward = jax.jit(partial(forward_integration, step_fn, n_spikes))
    loss_fn = partial(max_over_time_loss, forward)
    update_fn = partial(update, loss_fn)

    # init params
    rng = jax.random.PRNGKey(42)
    rng, trainset_rng, testset_rng = jax.random.split(rng, 3)
    weights = init_weights(rng, (n_input, n_hidden, n_output))

    testset = linear_dataset(testset_rng, 1_000)
    trainset = linear_dataset(trainset_rng, 1_000)

    def make_batches(dataset, batch_size):
        return (
            dataset[0].reshape(-1, batch_size, 4),
            dataset[1].reshape(-1, batch_size, 2),
        )

    test_loss, (test_acc, spikes, max_v) = jax.vmap(loss_fn, in_axes=(None, 0))(
        weights, testset
    )
    print(f"Loss: {np.mean(test_loss):.3f}, acc: {np.mean(test_acc):.3f}")

    # manually for debugging
    # loss_value = []
    # for i in range(len(testset[0])):
    #     weights, (loss, (acc, spikes, max_v)) = update(weights, (trainset[0][i], trainset[1][i]))
    #     loss_value.append(loss)

    # or with scan
    weights, (loss_value, (acc, spikes, max_v)) = jax.lax.scan(
        update_fn, weights, trainset
    )

    test_loss, (test_acc, _, _) = jax.vmap(loss_fn, in_axes=(None, 0))(weights, testset)
    print(f"Loss: {np.mean(test_loss):.3f}, acc: {np.mean(test_acc):.3f}")

    plt.plot(np.arange(len(loss_value)), loss_value)
    plt.show()
