from functools import partial
from typing import Callable, List, Optional, Tuple

import jax
import jax.numpy as np
import matplotlib.pyplot as plt
from jax import random

from jaxsnn.event.leaky_integrate_and_fire import LIFParameters, LIF, RecursiveLIF
from jaxsnn.event.serial import serial
from jaxsnn.event.root import ttfs_solver
from jaxsnn.types import Array, Spike, Weight


# A = np.array([[-tau_mem_inv, tau_mem_inv], [0, -tau_syn_inv]])


@jax.jit
def loss_fn_vec(t1, t2, target_1, target_2, tau_mem):
    loss_value = -(
        np.log(1 + np.exp(-np.abs(t1 - target_1) / tau_mem))
        + np.log(1 + np.exp(-np.abs(t2 - target_2) / tau_mem))
    )
    return loss_value


# loss over time
def plot_loss(ax, loss_value: Array):
    ax.plot(np.arange(len(loss_value)), loss_value, label="Loss")
    ax.title.set_text(f"Loss")


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
    ax.title.set_text(f"Output Spike Times")
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


@jax.jit
def log_loss(first_spikes: Array, target: Array, tau_mem: float):
    loss_value = -np.sum(np.log(1 + np.exp(-np.abs(first_spikes - target) / tau_mem)))
    return loss_value


def first_spike(spikes: Spike, size: int) -> Array:
    return np.array(
        [
            np.nanmin(np.where(spikes.idx == idx, spikes.time, np.nan))
            for idx in range(size)
        ]
    )


def loss(
    apply_fn: Callable,
    tau_mem: float,
    weights: List[Weight],
    batch: Tuple[Spike, Array],
) -> Tuple[float, Tuple[float, List[Spike]]]:

    input_spikes, target = batch
    recording = apply_fn(weights, input_spikes)
    output = recording[-1]
    size = weights[-1].shape[1]  # type: ignore
    t_first_spike = first_spike(output, size)

    return (log_loss(t_first_spike, target, tau_mem), (t_first_spike, recording))


def update(
    loss_fn: Callable,
    weights: List[Weight],
    batch: Tuple[Spike, Array],
):
    value, grad = jax.value_and_grad(loss_fn, has_aux=True)(weights, batch)
    # if any([(np.isnan(g)) for g in grad]):
    # assert False
    weights = jax.tree_map(lambda f, df: f - 0.1 * df, weights, grad)
    return weights, value


def train(trainset: Tuple[Spike, Array]):
    input_shape = 2

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

    apply_fn(weights, Spike(trainset[0].time[0], trainset[0].idx[0]))

    # declare update function
    update_fn = partial(update, partial(loss, apply_fn, tau_mem))

    # train the net
    weights, (loss_value, (output, recording)) = jax.lax.scan(
        update_fn, weights, trainset
    )

    # plot the results
    const_target = trainset[1][0]
    observe = np.arange(0, len(trainset[1]), len(trainset[1]) // 4)
    fix, (ax1, ax2, ax3) = plt.subplots(3, 4, figsize=(10, 8))
    plot_loss(ax1[0], loss_value)
    plot_output(ax1[1], output, t_max, const_target)
    plot_2dloss(ax1[2], output, t_max, const_target, 100, tau_mem)
    plot_spikes(ax2, recording[0], t_max, observe)
    plot_spikes(ax3, recording[1], t_max, observe, target=const_target)
    plt.show()


def constant_dataset(n_epochs, t_max):
    input_spikes = Spike(
        np.array([0.1, 0.2, 1]) * t_max,  # type: ignore
        np.array([0, 1, 0]),
    )
    target = np.array([0.2, 0.3]) * t_max  # type: ignore
    batch = (input_spikes, target)
    tiling = (n_epochs, 1)
    dataset = (
        Spike(np.tile(batch[0].time, tiling), np.tile(batch[0].idx, tiling)),
        np.tile(batch[1], tiling),
    )
    return dataset


if __name__ == "__main__":
    tau_mem = 1e-2
    tau_syn = 5e-3
    t_late = tau_syn + tau_mem
    t_max = 2 * t_late

    n_epochs = 2000
    trainset = constant_dataset(n_epochs, t_max)
    train(trainset)
