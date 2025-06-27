from functools import partial
import unittest
from typing import Callable, List, Tuple

import jax
import jax.numpy as np
from jax import random
from jaxsnn.base.params import LIFParameters
from jaxsnn.base.compose import serial
from jaxsnn.base.dataset import constant_dataset, data_loader
from jaxsnn.event.modules.leaky_integrate_and_fire import LIF
from jaxsnn.event.loss import loss_wrapper, target_time_loss
from jaxsnn.event.types import Spike, Weight, EventPropSpike


class TestEventTasksContant(unittest.TestCase):
    def update(
            self,
            loss_fn: Callable,
            weights: List[Weight],
            batch: Tuple[Spike, jax.Array]):
        value, grad = jax.value_and_grad(loss_fn, has_aux=True)(weights, batch)
        weights = jax.tree_map(lambda f, df: f - 0.1 * df, weights, grad)
        return weights, value

    def test_train(self):
        n_epochs = 2000
        input_shape = 2
        n_hidden = 10
        n_output = 2
        n_neurons = n_hidden + n_output

        params = LIFParameters()
        t_late = params.tau_syn + params.tau_mem
        t_max = 2 * t_late

        # declare net
        init_fn, apply_fn = serial(
            LIF(n_hidden,
                n_spikes=input_shape + n_hidden,
                t_max=t_max,
                params=params),
            LIF(n_output,
                n_spikes=input_shape + n_hidden + n_output,
                t_max=t_max,
                params=params))

        # init weights
        rng = random.PRNGKey(45)
        _, weights = init_fn(rng, input_shape)

        loss_fn = partial(
            loss_wrapper,
            apply_fn,
            target_time_loss,
            params.tau_mem,
            n_neurons,
            n_output,
            external=None,
            carry=None)
        update_fn = partial(self.update, loss_fn)

        # train the net
        trainset = constant_dataset(t_max, n_epochs)

        # Create Spikes from input
        spike_idx = np.array([0, 1, 0])
        input_spikes = EventPropSpike(
            trainset[0],
            np.tile(spike_idx, (n_epochs, 1)),
            np.zeros_like(trainset[0], dtype=trainset[0].dtype))
        trainset_encoded = (input_spikes, trainset[1])
        trainset_batched = data_loader(trainset_encoded, n_epochs)

        weights, (loss_value, _) = jax.lax.scan(
            update_fn, weights, trainset_batched)

        self.assertLess(loss_value[-1], -0.4)


if __name__ == '__main__':
    unittest.main()
