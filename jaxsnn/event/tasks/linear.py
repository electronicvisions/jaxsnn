from functools import partial
from typing import Callable, List, Optional, Tuple

import jax
import jax.numpy as np
import matplotlib.pyplot as plt
from jax import random

from jaxsnn.base.types import Array, Spike, Weight
from jaxsnn.event.compose import serial
from jaxsnn.event.dataset import linear_dataset
from jaxsnn.event.leaky_integrate_and_fire import LIF, LIFParameters, RecursiveLIF
from jaxsnn.event.loss import spike_time_loss
from jaxsnn.event.root import ttfs_solver


@jax.jit
def loss_fn_vec(t1, t2, target_1, target_2, tau_mem):
    loss_value = -(
        np.log(1 + np.exp(-np.abs(t1 - target_1) / tau_mem))
        + np.log(1 + np.exp(-np.abs(t2 - target_2) / tau_mem))
    )
    return loss_value


def plot_loss(ax, loss_value: Array):
    ax.plot(np.arange(len(loss_value)), loss_value, label="Loss")
    ax.title.set_text("Loss")


def plot_2dloss(
    ax, trajectory: Array, dt: float, target: Array, n: int, tau_mem: float
):
    t1 = np.linspace(0, 0.5, n)
    t2 = np.linspace(0, 0.5, n)
    xx, yy = np.meshgrid(t1, t2)
    zz = loss_fn_vec(xx, yy, target[0] / dt, target[1] / dt, tau_mem)

    ax.contourf(t1, t2, zz)
    ax.scatter(trajectory[:, 0] / dt, trajectory[:, 1] / dt, s=0.1, color="red")
    ax.axis("scaled")
    # ax.colorbar()


def plot_output(ax, t_output: Array, t_max: float, target: Array):
    ax.plot(np.arange(len(t_output)), t_output[:, 0] / t_max, label="t_spike 1")
    ax.plot(np.arange(len(t_output)), t_output[:, 1] / t_max, label="t_spike 2")
    ax.axhline(target[0] / t_max, color="red")
    ax.axhline(target[1] / t_max, color="red")
    ax.title.set_text("Spike Times Correct class")
    ax.legend()


def plot_spikes(
    axs, spikes: Spike, t_max: float, observe: Array, target: Optional[Array] = None
):
    for i, it in enumerate(observe):
        spike_times = spikes.time[it] / t_max
        s = 3 * (120.0 / len(spikes.time[it])) ** 2.0
        axs[i].scatter(x=spike_times, y=spikes.idx[it] + 1, s=s, marker="|", c="black")
        axs[i].set_ylabel("neuron id")
        axs[i].set_xlabel(r"$t$ [us]")
        if target is not None:
            axs[i].scatter(x=target[0] / t_max, y=1, s=s, marker="|", c="red")
            axs[i].scatter(x=target[1] / t_max, y=2, s=s, marker="|", c="red")


def update(
    loss_fn: Callable,
    learning_rate: float,
    weights: List[Weight],
    batch: Tuple[Spike, Array],
):
    value, grad = jax.value_and_grad(loss_fn, has_aux=True)(weights, batch)
    weights = jax.tree_map(lambda f, df: f - learning_rate * df, weights, grad)
    return weights, value


def loss_and_acc(
    loss_fn: Callable,
    weights: List[Weight],
    dataset: Tuple[Spike, Array],
):
    batched_loss = jax.vmap(loss_fn, in_axes=(None, 0))
    loss, (t_first_spike, _) = batched_loss(weights, dataset)
    accuracy = np.argmin(dataset[1], axis=1) == np.argmin(t_first_spike, axis=1)

    t_first_spike = np.where(t_first_spike == np.inf, np.nan, t_first_spike)
    first_target = t_first_spike[
        np.arange(len(dataset[1])), np.argmin(dataset[1], axis=1)
    ]
    second_target = t_first_spike[
        np.arange(len(dataset[1])), np.argmax(dataset[1], axis=1)
    ]
    return (
        np.mean(loss),
        np.mean(accuracy),
        np.nanmean(first_target),
        np.nanmean(second_target),
    )


def train(trainset: Tuple[Spike, Array], epochs):
    input_shape = 4

    learning_rate = 1e-2
    tau_mem = 1e-2
    tau_syn = 5e-3
    t_late = tau_syn + tau_mem
    t_max = 2 * t_late
    p = LIFParameters(tau_mem_inv=1 / tau_mem, tau_syn_inv=1 / tau_syn, v_th=0.6)
    solver = partial(ttfs_solver, tau_mem, p.v_th)

    # declare net
    init_fn, apply_fn = serial(
        t_max,
        RecursiveLIF(4, n_spikes=10, t_max=t_max, p=p, solver=solver),
        LIF(2, n_spikes=20, t_max=t_max, p=p, solver=solver),
    )

    # init weights
    rng = random.PRNGKey(42)
    weights = init_fn(rng, input_shape)

    # declare update function
    loss_fn = partial(spike_time_loss, apply_fn, tau_mem)
    update_fn = partial(update, loss_fn, learning_rate)

    def epoch(weights, _):
        weights, _ = jax.lax.scan(update_fn, weights, trainset)
        return weights, (loss_and_acc(loss_fn, weights, trainset), weights)

    # train the net
    weights, (res, weights_over_time) = jax.lax.scan(epoch, weights, np.arange(epochs))
    loss, acc, first_target, second_target = res

    fix, (ax1, ax2, ax3) = plt.subplots(3, 3, figsize=(8, 5))

    # Loss
    ax1[0].plot(np.arange(epochs), loss, label="Loss")
    ax1[0].set_xlabel("Epoch")
    ax1[0].title.set_text("Loss")

    # Accuracy
    ax1[1].plot(np.arange(epochs), acc, label="Accuracy")
    ax1[1].set_xlabel("Epoch")
    ax1[1].title.set_text("Accuracy")

    # Avg spike time correct neuron
    ax1[2].plot(np.arange(epochs), first_target / t_max, label="t_spike correct neuron")
    ax1[2].axhline(np.min(trainset[1]) / t_max, color="red")

    # plot spike time non-correct neuron
    ax1[2].plot(np.arange(epochs), second_target / t_max, label="t_spike false neuron")
    ax1[2].axhline(np.max(trainset[1]) / t_max, color="red")

    ax1[2].set_xlabel("Epoch")
    ax1[2].title.set_text("Output spike times")
    ax1[2].legend()

    # run again
    batched_loss = jax.vmap(loss_fn, in_axes=(None, 0))
    _, (t_first_spike, _) = batched_loss(weights, trainset)

    predicted_class = np.argmin(t_first_spike, axis=1)
    x = trainset[0].time[:, 0]
    y = trainset[0].time[:, 1]
    ax2[0].scatter(x, y, s=10, c=predicted_class)
    ax2[0].title.set_text("Predicted class")

    correct_class = np.argmin(trainset[1], axis=1)
    x = trainset[0].time[:, 0]
    y = trainset[0].time[:, 1]
    ax2[1].scatter(x, y, s=10, c=correct_class)
    ax2[1].title.set_text("Correct class")

    input_weights = weights_over_time[0][0].reshape(100, -1)
    for i in range(input_weights.shape[1]):
        ax3[0].plot(np.arange(epochs), input_weights[:, i])
        ax3[0].title.set_text("Input weights")

    recursive_weights = weights_over_time[0][1].reshape(100, -1)
    for i in range(recursive_weights.shape[1]):
        ax3[1].plot(np.arange(epochs), recursive_weights[:, i])
        ax3[1].title.set_text("Recursive weights")

    output_weights = weights_over_time[1].reshape(100, -1)
    for i in range(output_weights.shape[1]):
        ax3[2].plot(np.arange(epochs), output_weights[:, i])
        ax3[2].title.set_text("Output weights")

    # TODO look at activity, when input, when internal, when output
    # add batching
    # look at loss function
    # look at LI neuron in output layer
    # observe = [0, 30, 60, 90]
    # plot_spikes(ax2, recording[0], t_max, observe)
    # plot_spikes(ax3, recording[1], t_max, observe, target=const_target)
    plt.show()


if __name__ == "__main__":
    tau_mem = 1e-2
    tau_syn = 5e-3
    t_late = tau_syn + tau_mem
    t_max = 2 * t_late

    n_samples = 1000
    epochs = 100
    rng = random.PRNGKey(42)
    trainset = linear_dataset(rng, t_max, [n_samples])
    train(trainset, epochs)
