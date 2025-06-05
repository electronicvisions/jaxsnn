from typing import (
    Any,
    Callable,
    Dict,
    Tuple,
)
import unittest

from functools import partial

import jax
import jax.numpy as jnp
from jax import random

from jaxsnn.base.dataset import (
    constant_dataset,
    data_loader,
)
from jaxsnn.event.modules import LIFParameters
from jaxsnn.event.loss import target_time_loss
from jaxsnn.event.types import (
    Spike,
    Parameters,
)
from jaxsnn.event.modules import (
    LIF,
    Linear,
    Source,
)
from jaxsnn.event.topology import Topology
from jaxsnn.event.loss import first_spike


class TestEventTasksContant(unittest.TestCase):

    def loss_wrapper(
        self,
        apply_fn,
        loss_fn: Callable[[jax.Array, jax.Array, float], float],
        tau_mem: float,
        n_outputs: int,
        output_layer: str,
        weights: Parameters,
        batch: Tuple[Spike, jax.Array],
    ) -> Tuple[jax.Array, Tuple[jax.Array, Any]]:
        input_spikes, target = batch

        first_spike_function = jax.vmap(first_spike, in_axes=(0, None))
        loss_function = jax.vmap(loss_fn, in_axes=(0, 0, None))

        output = apply_fn(input_spikes, weights)

        t_first_spike = first_spike_function(
            output[output_layer],
            n_outputs,
        )
        loss_value = jnp.mean(loss_function(t_first_spike, target, tau_mem))

        return loss_value, (t_first_spike, None)

    def update(
        self,
        loss_fn: Callable,
        weights: Dict[str, jax.Array],
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

        params = LIFParameters(v_reset=-1000.0)
        t_late = params.tau_syn + params.tau_mem
        t_max = 2 * t_late

        builder = Topology(t_max=t_max)

        builder.add(
            {
                "inp": Source(input_shape),
                "lif1": LIF(
                    n_hidden,
                    n_hidden + input_shape,
                    params,
                ),
                "lif2": LIF(
                    n_output,
                    input_shape + n_output + n_hidden,
                    params,
                ),
                "syn1": Linear(
                    mean=0.8,
                    std=0.8,
                    min_delay=0.000,
                ),
                "syn2": Linear(
                    mean=0.9,
                    std=0.1,
                    min_delay=0.000,
                ),
            }
        )

        # connect modules
        builder.connect(
            [
                ("inp", "syn1"),
                ("syn1", "lif1"),
                ("lif1", "syn2"),
                ("syn2", "lif2"),
            ]
        )

        # declare net
        init_fn, apply_fn = builder.done()

        # init weights
        rng = random.PRNGKey(45)
        weights = init_fn(rng)

        loss_fn = partial(
            self.loss_wrapper,
            apply_fn,
            target_time_loss,
            params.tau_mem,
            n_output,
            "lif2",
        )

        # train the net
        trainset = constant_dataset(t_max, n_epochs)

        # Create Spikes from input
        spike_idx = jnp.array([0, 1, 0])
        input_spikes = Spike(
            time=trainset[0],
            current=jnp.zeros_like(trainset[0], dtype=trainset[0].dtype),
            idx=jnp.tile(spike_idx, (n_epochs, 1)),
            layer_idx=jnp.zeros_like(trainset[0], dtype=int),
            internal=jnp.ones_like(trainset[0], dtype=bool),
        )

        trainset_encoded = ({"inp": input_spikes}, trainset[1])
        trainset_batched = data_loader(trainset_encoded, n_epochs)

        weights, (loss_value, _) = jax.lax.scan(
            partial(
                self.update,
                loss_fn,
            ),
            weights,
            trainset_batched,
        )

        self.assertLess(loss_value[-1], -0.4)


if __name__ == '__main__':
    unittest.main()
