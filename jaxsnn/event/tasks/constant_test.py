from functools import partial

import jax
from jax import random

from jaxsnn.event.compose import serial
from jaxsnn.event.leaky_integrate_and_fire import LIF, LIFParameters, RecursiveLIF
from jaxsnn.event.root import ttfs_solver
from jaxsnn.event.tasks.constant import update
from jaxsnn.event.loss import target_time_loss, loss_wrapper
from jaxsnn.event.dataset import constant_dataset


def test_train():
    n_epochs = 2000
    input_shape = 2

    p = LIFParameters()
    t_late = p.tau_syn + p.tau_mem
    t_max = 2 * t_late
    solver = partial(ttfs_solver, p.tau_mem, p.v_th)

    # declare net
    init_fn, apply_fn = serial(
        RecursiveLIF(4, n_spikes=10, t_max=t_max, p=p, solver=solver),
        LIF(2, n_spikes=20, t_max=t_max, p=p, solver=solver),
    )

    # init weights
    rng = random.PRNGKey(42)
    weights = init_fn(rng, input_shape)

    loss_fn = partial(loss_wrapper, apply_fn, target_time_loss, p.tau_mem)
    update_fn = partial(update, loss_fn)

    # train the net
    trainset = constant_dataset(t_max, [n_epochs])
    weights, (loss_value, _) = jax.lax.scan(update_fn, weights, trainset[:2])
    assert loss_value[-1] <= -0.4
