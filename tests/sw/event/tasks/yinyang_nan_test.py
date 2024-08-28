import jax
import jax.numpy as np
from jax import random
from jaxsnn.base.params import LIFParameters
from jaxsnn.event.compose import serial
from jaxsnn.event.dataset.yinyang import yinyang_dataset
from jaxsnn.event.leaky_integrate_and_fire import LIF
from jaxsnn.event.types import EventPropSpike
from jaxsnn.event.utils import load_weights
import unittest


class TestEventTasksYinYangNan(unittest.TestCase):
    def test_nans(self):
        # in this example, nothing spikes in the first and second layer
        # this leads to nan gradients for the first layer and nan gradients for the second layer
        params = LIFParameters(v_reset=-1000.0)
        t_late = 2.0 * params.tau_syn
        t_max = 2.0 * t_late

        train_samples = 6400
        batch_size = 16
        n_train_batches = int(train_samples / batch_size)
        t_bias = 0.0

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
            t_bias=t_bias,
            t_correct_target=0.9 * t_late,
            t_wrong_target=1.5 * t_late,
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

        weights = load_weights(["tests/sw/event/tasks/weights7.npy"])

        # declare net
        _, apply_fn = serial(LIF(hidden_size, n_spikes_hidden, t_max, params))

        def loss_fn(weights, input_spikes):
            _, _, _, recording = apply_fn(weights, input_spikes, None, None)
            return recording[0].time[3], recording

        loss_fn(weights, batch[0])
        (loss, recording), grad = jax.value_and_grad(loss_fn, has_aux=True)(
            weights, batch[0]
        )
        assert not np.isnan(np.mean(grad[0].input))


if __name__ == '__main__':
    unittest.main()
