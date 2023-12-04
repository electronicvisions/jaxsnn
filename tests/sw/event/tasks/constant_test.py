from functools import partial
from typing import Callable, List, Tuple

import jax
from jax import random
from jaxsnn.base.params import LIFParameters
from jaxsnn.event.compose import serial
from jaxsnn.event.dataset import constant_dataset
from jaxsnn.event.leaky_integrate_and_fire import LIF
from jaxsnn.event.loss import loss_wrapper, target_time_loss
from jaxsnn.event.types import Spike, Weight
import unittest


class TestEventTasksContant(unittest.TestCase):
    def update(
        self,
        loss_fn: Callable,
        weights: List[Weight],
        batch: Tuple[Spike, jax.Array],
    ):
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
            LIF(
                n_hidden,
                n_spikes=input_shape + n_hidden,
                t_max=t_max,
                params=params,
            ),
            LIF(
                n_output,
                n_spikes=input_shape + n_hidden + n_output,
                t_max=t_max,
                params=params,
            ),
        )

        # init weights
        rng = random.PRNGKey(42)
        weights = init_fn(rng, input_shape)

        loss_fn = partial(
            loss_wrapper,
            apply_fn,
            target_time_loss,
            params.tau_mem,
            n_neurons,
            n_output,
        )
        update_fn = partial(self.update, loss_fn)

        # train the net
        trainset = constant_dataset(t_max, [n_epochs])
        weights, (loss_value, _) = jax.lax.scan(update_fn, weights, trainset[:2])
        assert loss_value[-1] <= -0.4, loss_value


if __name__ == '__main__':
    unittest.main()
