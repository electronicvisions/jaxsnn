from functools import partial
from typing import Callable, List, Optional, Tuple

import jax
import jax.numpy as np
import matplotlib.pyplot as plt
from jax import random

from jaxsnn.event.leaky_integrate_and_fire import LIFParameters, LIF
from jaxsnn.event.compose import serial
from jaxsnn.event.root import ttfs_solver
from jaxsnn.base.types import Array, Spike, Weight
from jaxsnn.event.dataset import constant_dataset, Dataset
from jaxsnn.event.loss import target_time_loss, loss_wrapper


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
    ax.title.set_text("Output Spike Times")
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
    weights: List[Weight],
    batch: Tuple[Spike, Array],
):
    value, grad = jax.value_and_grad(loss_fn, has_aux=True)(weights, batch)
    weights = jax.tree_map(lambda f, df: f - 0.1 * df, weights, grad)
    return weights, value


def train(trainset: Dataset):
    input_shape = 2

    p = LIFParameters()
    t_late = p.tau_syn + p.tau_mem
    t_max = 2 * t_late
    solver = partial(ttfs_solver, p.tau_mem, p.v_th)

    n_hidden = 4
    n_output = 2
    n_neurons = n_hidden + n_output

    # declare net
    init_fn, apply_fn = serial(
        LIF(4, n_spikes=10, t_max=t_max, p=p, solver=solver),
        LIF(2, n_spikes=20, t_max=t_max, p=p, solver=solver),
    )

    # init weights
    rng = random.PRNGKey(42)
    weights = init_fn(rng, input_shape)

    # declare update function
    loss_fn = partial(
        loss_wrapper, apply_fn, target_time_loss, p.tau_mem, n_neurons, n_output
    )
    update_fn = partial(update, loss_fn)

    # train the net
    weights, (loss_value, (output, recording)) = jax.lax.scan(
        update_fn, weights, trainset[:2]
    )

    # plot the results
    const_target = trainset[1][0]
    observe = np.arange(0, len(trainset[1]), len(trainset[1]) // 4)
    fix, (ax1, ax2, ax3) = plt.subplots(3, 4, figsize=(10, 8))
    plot_loss(ax1[0], loss_value)
    plot_output(ax1[1], output, t_max, const_target)
    plot_2dloss(ax1[2], output, t_max, const_target, 100, p.tau_mem)
    plot_spikes(ax2, recording[0], t_max, observe)
    plot_spikes(ax3, recording[1], t_max, observe, target=const_target)
    plt.show()


if __name__ == "__main__":
    p = LIFParameters()
    t_max = 2 * (p.tau_syn + p.tau_mem)

    n_epochs = 2000
    trainset = constant_dataset(t_max, [n_epochs])
    train(trainset)
