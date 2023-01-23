import time
from functools import partial
from typing import List, Tuple

import jax
import numpy as np
import optax
from jax import random

from jaxsnn.base.types import Array, Spike, Weight
from jaxsnn.event.compose import serial
from jaxsnn.event.dataset.toy import linear_dataset
from jaxsnn.event.functional import batch_wrapper
from jaxsnn.event.leaky_integrate_and_fire import LIF, LIFParameters, RecursiveLIF
from jaxsnn.event.loss import target_time_loss
from jaxsnn.event.root import ttfs_solver


def bench():
    # neuron params
    tau_mem = 1e-2
    tau_syn = 5e-3
    t_max = 6 * tau_syn
    v_th = 0.6
    p = LIFParameters(tau_mem_inv=1 / tau_mem, tau_syn_inv=1 / tau_syn, v_th=v_th)

    # training params
    step_size = 1e-3
    n_batches = 100
    batch_size = 32
    epochs = 50

    # net
    hidden_size = 4
    output_size = 2
    n_spikes_hidden = 20
    n_spikes_output = 30
    seed = 42
    optimizer_fn = optax.adam

    rng = random.PRNGKey(seed)
    param_rng, train_rng = random.split(rng, 2)
    trainset = linear_dataset(train_rng, tau_syn, [n_batches, batch_size])
    input_size = trainset[0].idx.shape[-1]
    solver = partial(ttfs_solver, tau_mem, p.v_th)

    # declare net
    init_fn, apply_fn = serial(
        RecursiveLIF(
            hidden_size, n_spikes=n_spikes_hidden, t_max=t_max, p=p, solver=solver
        ),
        LIF(output_size, n_spikes=n_spikes_output, t_max=t_max, p=p, solver=solver),
    )

    # init params and optimizer
    params = init_fn(param_rng, input_size)

    optimizer = optimizer_fn(step_size)
    opt_state = optimizer.init(params)

    # declare update function
    loss_fn = batch_wrapper(partial(target_time_loss, apply_fn, tau_mem))

    # define update function
    def update(
        input: Tuple[optax.OptState, List[Weight]],
        batch: Tuple[Spike, Array],
    ):
        opt_state, params = input
        value, grad = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        return (opt_state, params), value

    def epoch(state, _):
        state, _ = jax.lax.scan(update, state, trainset)
        return state, _

    # let it compile
    jax.lax.scan(epoch, (opt_state, params), np.arange(epochs))

    start = time.time()
    params, (loss_value, _) = jax.lax.scan(
        epoch, (opt_state, params), np.arange(epochs)
    )
    duration = time.time() - start
    print(f"{(1e3 * duration / epochs):.1f} ms per epoch")
    print(f"{(1e6 * duration / (epochs * n_batches)):.1f} µs per batch")
    print(f"{(1e6 * duration / (epochs * n_batches * batch_size)):.2f} µs per sample")


if __name__ == "__main__":
    bench()
