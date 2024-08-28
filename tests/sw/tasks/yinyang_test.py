from functools import partial

import jax
import optax
from jax import random
from jaxsnn.discrete.compose import serial
from jaxsnn.discrete.leaky_integrate import LI
from jaxsnn.discrete.leaky_integrate_and_fire import LIF
from jaxsnn.discrete.decode import max_over_time_decode
from jaxsnn.discrete.encode import spatio_temporal_encode
from jaxsnn.base.params import LIFParameters
from jaxsnn.discrete.dataset.yinyang import YinYangDataset, data_loader
from jaxsnn.discrete.loss import acc_and_loss, nll_loss
from jaxsnn.discrete.threshold import superspike
import unittest


class TestTasksYinYang(unittest.TestCase):
    def update(self, optimizer, state, batch, loss_fn):
        opt_state, weights, i = state
        input, output = batch

        (loss, recording), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            weights, (input, output), max_over_time_decode
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        weights = optax.apply_updates(weights, updates)
        return (opt_state, weights, i + 1), recording

    def test_train(self):
        n_classes = 3
        input_shape = 5
        batch_size = 64
        epochs = 3
        bias_spike = 0.0

        hidden_features = 70
        expected_spikes = 0.5
        step_size = 1e-3
        DT = 5e-4

        t_late = (
            1.0 / LIFParameters().tau_syn_inv + 1.0 / LIFParameters().tau_mem_inv
        )
        time_steps = int(2 * t_late / DT)

        # define train and test data
        rng = random.PRNGKey(42)
        rng, train_key, test_key, init_key = random.split(rng, 4)
        trainset = YinYangDataset(train_key, 4992, bias_spike=bias_spike)
        test_dataset = YinYangDataset(test_key, 1000, bias_spike=bias_spike)

        trainset_batches = data_loader(trainset, batch_size, None)
        # Encoding the inputs
        input_encoder_batched = jax.vmap(
            spatio_temporal_encode,
            in_axes=(0, None, None, None)
            )
        train_input_encoded = input_encoder_batched(
            trainset_batches[0],
            time_steps,
            t_late,
            DT,
        )
        trainset = (train_input_encoded, trainset_batches[1])

        test_input_encoded = spatio_temporal_encode(
            test_dataset.vals,
            time_steps,
            t_late,
            DT,
        )
        test_dataset.vals = test_input_encoded

        # define the network
        snn_init, snn_apply = serial(
            LIF(hidden_features),
            LI(n_classes),
        )

        _, weights = snn_init(init_key, input_shape=input_shape)

        optimizer = optax.adam(step_size)
        opt_state = optimizer.init(weights)

        # define functions
        snn_apply = partial(snn_apply, recording=True)
        loss_fn = partial(nll_loss, snn_apply, expected_spikes=expected_spikes)
        train_step_fn = partial(partial(self.update, optimizer), loss_fn=loss_fn)

        for _ in range(epochs):
            (opt_state, weights, _), _ = jax.lax.scan(
                train_step_fn, (opt_state, weights, 0), trainset
            )

        accuracy, _ = acc_and_loss(
            snn_apply,
            weights,
            (test_dataset.vals, test_dataset.classes),
            max_over_time_decode
        )
        assert accuracy > 0.70


if __name__ == '__main__':
    unittest.main()
