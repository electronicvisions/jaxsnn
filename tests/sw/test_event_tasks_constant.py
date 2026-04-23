from functools import partial
import unittest
from typing import Any, Callable, List, Tuple, Optional

import jax
import jax.numpy as jnp
from jax import random
from jaxsnn.base.params import LIFParameters
from jaxsnn.base.compose import serial
from jaxsnn.base.dataset import constant_dataset, data_loader
from jaxsnn.event.modules.leaky_integrate_and_fire import LIF
from jaxsnn.event.loss import target_time_loss, first_spike
from jaxsnn.event.types import (
    Apply,
    EventPropSpike,
    Weight,
    Spike,
)


class TestEventTasksContant(unittest.TestCase):

    def loss_wrapper(
            self,
            apply_fn: Apply,
            loss_fn: Callable[[jax.Array, jax.Array, float], float],
            tau_mem: float,
            n_neurons: int,
            n_outputs: int,
            weights: List[Weight],
            batch: Tuple[Spike, jax.Array],
        ) -> Tuple[jax.Array, Tuple[jax.Array, Any]]:
        input_spikes, target = batch

        apply_fn = jax.vmap(apply_fn, in_axes=(None, 0, None, None))
        first_spike_function = jax.vmap(first_spike, in_axes=(0, None, None))
        loss_function = jax.vmap(loss_fn, in_axes=(0, 0, None))

        _, _, output, recording = apply_fn(
            weights,
            input_spikes,
            None,
            None,
        )

        t_first_spike = first_spike_function(output, n_neurons, n_outputs)
        loss_value = jnp.mean(loss_function(t_first_spike, target, tau_mem))

        return loss_value, (t_first_spike, recording)

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
            self.loss_wrapper,
            apply_fn,
            target_time_loss,
            params.tau_mem,
            n_neurons,
            n_output,
        )
        update_fn = partial(self.update, loss_fn)

        # train the net
        trainset = constant_dataset(t_max, n_epochs)

        # Create Spikes from input
        spike_idx = jnp.array([0, 1, 0])
        input_spikes = EventPropSpike(
            trainset[0],
            jnp.tile(spike_idx, (n_epochs, 1)),
            jnp.zeros_like(trainset[0], dtype=trainset[0].dtype))
        trainset_encoded = (input_spikes, trainset[1])
        trainset_batched = data_loader(trainset_encoded, n_epochs)

        weights, (loss_value, _) = jax.lax.scan(
            update_fn, weights, trainset_batched)

        self.assertLess(loss_value[-1], -0.4)


if __name__ == '__main__':
    unittest.main()
