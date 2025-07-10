import jax
import jax.numpy as jnp
from jax import random
from jaxsnn.base.params import LIFParameters
from jaxsnn.base.compose import serial
from jaxsnn.base.dataset import yinyang_dataset, data_loader
from jaxsnn.event.encode import spatio_temporal_encode, target_temporal_encode
from jaxsnn.event.modules.leaky_integrate_and_fire import LIF
from jaxsnn.event.types import EventPropSpike
from jaxsnn.event.utils import load_weights
import unittest


class TestEventTasksYinYangNan(unittest.TestCase):
    def test_nans(self):
        # in this example, nothing spikes in the first and second layer
        # this leads to nan gradients for the first layer and nan gradients for
        # the second layer
        params = LIFParameters(v_reset=-1000.0)
        t_late = 2.0 * params.tau_syn
        t_max = 2.0 * t_late

        train_samples = 6400
        bias_spike = 0.0

        # net
        hidden_size = 60
        n_spikes_hidden = 4

        rng = random.PRNGKey(42)
        trainset = yinyang_dataset(rng, train_samples, True, bias_spike)

        # Encoding
        correct_target_time = 0.9 * params.tau_syn
        wrong_target_time = 1.1 * params.tau_syn
        n_classes = 3
        target_encoding_params = [
            correct_target_time,
            wrong_target_time,
            n_classes
        ]

        input_encoder_batched = jax.vmap(
            spatio_temporal_encode, in_axes=(0, None, None, None))
        target_encoder_batched = jax.vmap(
            target_temporal_encode, in_axes=(0, None, None, None))

        train_input_encoded = input_encoder_batched(
            trainset[0], t_late, None, False)
        train_targets_encoded = target_encoder_batched(
            trainset[1], *target_encoding_params,)

        trainset = data_loader(
            (train_input_encoded, train_targets_encoded), 64)

        bad_idx = 79
        i = 1
        batch = (
            EventPropSpike(
                trainset[0].time[bad_idx][i],
                trainset[0].idx[bad_idx][i],
                jnp.zeros_like(trainset[0].time[bad_idx][i]),
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
            weights, batch[0])

        self.assertFalse(jnp.isnan(jnp.mean(grad[0].input)))


if __name__ == '__main__':
    unittest.main()
