from functools import partial

import jax
import jax.numpy as np
import optax
from jax import random
from jaxsnn.base.compose import serial
from jaxsnn.discrete.leaky_integrate import LI
from jaxsnn.discrete.leaky_integrate_and_fire import LIF
from jaxsnn.discrete.decode import max_over_time_decode
from jaxsnn.discrete.encode import spatio_temporal_encode
from jaxsnn.base.params import LIFParameters
from jaxsnn.base.dataset import yinyang_dataset, data_loader
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
        input_size = 5
        batch_size = 64
        epochs = 4
        n_train_batches = 78
        train_samples = batch_size * n_train_batches
        test_samples = 1000

        bias_spike = 0.0
        mirror = True

        hidden_features = 70
        expected_spikes = 0.5
        step_size = 1e-3
        DT = 5e-4

        t_late = LIFParameters().tau_syn + LIFParameters().tau_mem
        time_steps = int(2 * t_late / DT)

        # Define random keys
        rng = random.PRNGKey(42)
        init_key, train_key, test_key, shuffle_key = random.split(rng, 4)
        
        # Setting up trainset and testset
        trainset = yinyang_dataset(train_key, train_samples, mirror, bias_spike)
        testset = yinyang_dataset(test_key, test_samples, mirror, bias_spike)

        # Encoding the inputs
        input_encoder_batched = jax.vmap(
            spatio_temporal_encode,
            in_axes=(0, None, None, None)
        )
        train_input_encoded = input_encoder_batched(
            trainset[0],
            time_steps,
            t_late,
            DT,
        )
        trainset = (train_input_encoded, trainset[1])

        test_input_encoded = input_encoder_batched(
            testset[0],
            time_steps,
            t_late,
            DT,
        )
        testset = (test_input_encoded, testset[1])

        # define the network
        snn_init, snn_apply = serial(
            LIF(hidden_features),
            LI(n_classes),
        )

        _, weights = snn_init(init_key, input_size)

        optimizer = optax.adam(step_size)
        opt_state = optimizer.init(weights)

        # define functions
        loss_fn = partial(nll_loss, snn_apply, expected_spikes=expected_spikes)
        train_step_fn = partial(partial(self.update, optimizer), loss_fn=loss_fn)

        for _ in range(epochs):
            # Generate randomly shuffled batches
            this_shuffle_key, shuffle_key = random.split(shuffle_key)
            trainset_batched = data_loader(trainset, 64, this_shuffle_key)

            # Swap axes because time axis needs to come before batch axis
            trainset_batched = (
                np.swapaxes(trainset_batched[0], 1, 2),
                trainset_batched[1]
            )
            (opt_state, weights, _), _ = jax.lax.scan(
                train_step_fn, (opt_state, weights, 0), trainset_batched
            )

        # Implementation requires time axis to come before batch axis
        testset = (
            np.swapaxes(testset[0], 0, 1),
            testset[1]
        )

        accuracy, _ = acc_and_loss(
            snn_apply,
            weights,
            (testset[0], testset[1]),
            max_over_time_decode
        )
        assert accuracy > 0.70


if __name__ == '__main__':
    unittest.main()
