from functools import partial

import jax
from jax import random

from jaxsnn.event.compose import serial
from jaxsnn.event.leaky_integrate_and_fire import LIF, LIFParameters, RecursiveLIF
from jaxsnn.event.root import ttfs_solver
from jaxsnn.event.tasks.constant import update
from jaxsnn.event.loss import spike_time_loss
from jaxsnn.event.dataset import constant_dataset


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
    loss_fn = partial(spike_time_loss, apply_fn, tau_mem)
    update_fn = partial(update, loss_fn)

    # train the net
    trainset = constant_dataset(t_max, [n_epochs])
    weights, (loss_value, _) = jax.lax.scan(update_fn, weights, trainset)
    assert loss_value[-1] <= -1.3
