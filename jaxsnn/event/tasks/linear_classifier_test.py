from functools import partial
from typing import Callable, List, Tuple

import jax
import jax.numpy as np
from jax import random

from jaxsnn.event.leaky_integrate_and_fire import LIFParameters, LIF, RecursiveLIF
from jaxsnn.event.compose import serial
from jaxsnn.event.root import ttfs_solver
from jaxsnn.base.types import Array, Spike, Weight


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
    weights = jax.tree_map(lambda f, df: f - 0.1 * df, weights, grad)
    return weights, value


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


def test_train():
    n_epochs = 2000
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

    # declare update function
    loss_fn = partial(loss, apply_fn, tau_mem)
    update_fn = partial(update, loss_fn)

    # train the net
    trainset = constant_dataset(n_epochs, t_max)
    weights, (loss_value, _) = jax.lax.scan(update_fn, weights, trainset)
    assert loss_value[-1] <= -1.3
