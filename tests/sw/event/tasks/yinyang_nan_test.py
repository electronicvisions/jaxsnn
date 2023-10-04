import jax
import jax.numpy as np
from jax import random
from jaxsnn.base.params import LIFParameters
from jaxsnn.event.compose import serial
from jaxsnn.event.dataset.yinyang import yinyang_dataset
from jaxsnn.event.leaky_integrate_and_fire import LIF
from jaxsnn.event.types import EventPropSpike
from jaxsnn.event.utils import load_params


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

    seed = 42

    rng = random.PRNGKey(seed)
    param_rng, train_rng, test_rng = random.split(rng, 3)
    trainset = yinyang_dataset(
        train_rng,
        [n_train_batches, batch_size],
        t_late,
        bias_spike=bias_spike,
        correct_target_time=0.9 * t_late,
        wrong_target_time=1.5 * t_late,
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

    params = load_params(["tests/sw/event/tasks/weights7.npy"])

    # declare net
    _, apply_fn = serial(LIF(hidden_size, n_spikes_hidden, t_max, p))

    def loss_fn(weights, input_spikes):
        recording = apply_fn(weights, input_spikes)
        return recording[0].time[3], recording

    loss_fn(params, batch[0])
    (loss, recording), grad = jax.value_and_grad(loss_fn, has_aux=True)(
        params, batch[0]
    )
    assert not np.isnan(np.mean(grad[0].input))
