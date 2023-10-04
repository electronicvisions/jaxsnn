from functools import partial

import jax
import jax.numpy as np
from jax import random

from jaxsnn.base.types import EventPropSpike
from jaxsnn.event.compose import serial
from jaxsnn.event.dataset.toy import yinyang_dataset
from jaxsnn.event.leaky_integrate_and_fire import LIF, LIFParameters
from jaxsnn.event.root import ttfs_solver
from jaxsnn.event.utils import load_params

from jax.config import config


config.update("jax_debug_nans", True)


def test_nans():
    # in this example, nothing spikes in the first and second layer
    # this leads to nan gradients for the first layer and nan gradients for the second layer
    p = LIFParameters(v_reset=-1000.0)
    t_late = 2.0 * p.tau_syn
    t_max = 2.0 * t_late

    train_samples = 6400
    batch_size = 16
    n_train_batches = int(train_samples / batch_size)
    bias_spike = 0.0

    # net
    hidden_size = 60
    n_spikes_hidden = 4

    solver = partial(ttfs_solver, p.tau_mem, p.v_th)
    seed = 42

    rng = random.PRNGKey(seed)
    param_rng, train_rng, test_rng = random.split(rng, 3)
    trainset = yinyang_dataset(
        train_rng,
        t_late,
        [n_train_batches, batch_size],
        bias_spike=bias_spike,
    )

    bad_idx = 79
    i = 1
    batch = (
        EventPropSpike(
            trainset[0].time[bad_idx][i],
            trainset[0].idx[bad_idx][i],
            np.zeros_like(trainset[0].time[bad_idx][i]),
        ),
        trainset[1][bad_idx][i],
    )

    params = load_params(["src/pyjaxsnn/jaxsnn/event/tasks/weights7.npy"])

    # declare net
    _, apply_fn = serial(
        LIF(
            hidden_size,
            n_spikes_hidden,
            t_max,
            p,
            solver,
        ),
    )
    # apply_fn = jax.jit(apply_fn)

    def loss_fn(
        weights,
        input_spikes,
    ):
        recording = apply_fn(weights, input_spikes)
        return recording[0].time[3], recording

    loss_fn(params, batch[0])
    (loss, recording), grad = jax.value_and_grad(loss_fn, has_aux=True)(
        params, batch[0]
    )
    assert not np.isnan(np.mean(grad[0].input))
