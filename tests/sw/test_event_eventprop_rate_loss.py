import unittest

import numpy as np
import jax
from jax import random
import jax.numpy as jnp

from jaxsnn.event.types import Spike
from jaxsnn.event.loss import rate_loss
from jaxsnn.event.modules import Source, LIF, LIFParameters, Linear
from jaxsnn.event.topology import Topology


class TestEventPropRateLoss(unittest.TestCase):

    input_size = 25
    t_max = 100e-6

    def setUp(self):
        """Set up test fixtures."""
        self.rng = jax.random.PRNGKey(42)
        self.tau_mem = 20.0

    def test_rate_loss_analytical(self):
        """ Test rate loss gradient computed analytically """
        lif_params = LIFParameters(
            v_th=0.6,
            v_leak=0.0,
            v_reset=0.0,
            tau_syn=10e-6,
            tau_mem=20e-6,
        )

        # define trainset and testset
        rng = random.PRNGKey(np.random.randint(0, 10000))
        param_rng, data_rng, rng = random.split(rng, 3)

        spike_times = jnp.sort(
            random.uniform(data_rng, (1, self.input_size), maxval=self.t_max)
        )
        spike_rng, idx_rng = random.split(data_rng)
        spike_times = jnp.sort(
            random.uniform(spike_rng, (1, self.input_size), maxval=self.t_max)
        )
        spike_idx = random.randint(
            idx_rng, (1, self.input_size), 0, self.input_size,
        )

        inputs = {
            "inp": Spike(
                time=spike_times,
                idx=spike_idx,
                internal=jnp.ones((1, self.input_size), dtype=bool),
                layer_idx=jnp.zeros((1, self.input_size), dtype=jnp.int32),
                current=jnp.zeros((1, self.input_size)),
            )
        }

        # define topology
        builder = Topology(
            mock=True,
            t_max=50e-6,
            backprop_method="eventprop",
        )

        # create modules
        builder.add(
            {
                "inp": Source(
                    size=self.input_size,
                ),
                "lif": LIF(
                    size=50,
                    n_steps=300,
                    params=lif_params,
                    compute_rate=True,
                ),
                "syn": Linear(
                    mean=0.8,
                    std=0.5,
                    min_delay=0.0,
                ),
            }
        )

        # connect modules
        builder.connect(
            [
                ("inp", "syn"),
                ("syn", "lif"),
            ]
        )

        init_fn, apply_fn = builder.done()

        # init weights
        params = init_fn(param_rng)

        observables = apply_fn(
            inputs,
            params,
        )

        for idx in range(50):
            self.assertEqual(
                jnp.count_nonzero(
                    observables["lif"].spike.get_internal().idx == idx,
                ),
                observables["lif"].rate[0, idx]
            )

        target_count = 10.0

        def loss_fn(weights, inputs):
            observables = apply_fn(
                inputs,
                weights,
            )
            spike_count = observables["lif"].rate
            target_counts = jnp.ones_like(spike_count) * target_count
            count_factor = 1.0
            rloss = rate_loss(
                spike_count,
                target_counts,
                count_factor,
            )
            return rloss, observables

        # Backward
        grad, res = jax.grad(
            loss_fn,
            has_aux=True,
        )(
            params,
            inputs,
        )

        # Verify gradients on the correct entries
        for i in range(50):
            for j in range(self.input_size):
                gradient = grad["syn"][j, i]
                # Check if input spike
                has_input_spike = jnp.any(inputs["inp"].idx[0, :] == j)
                has_output_spike = jnp.any(res["lif"].spike.get_internal().idx[0, :] == i)

                if not has_input_spike or not has_output_spike:
                    self.assertEqual(gradient, 0.0)
                    continue

                # Rate loss has zero slope exactly at the target count, so
                # the gradient is zero regardless of timing.
                if res["lif"].rate[0, i] == target_count:
                    self.assertEqual(gradient, 0.0)
                    continue

                spike_time_j = jnp.min(
                    inputs["inp"].time[0, inputs["inp"].idx[0, :] == j]
                )
                spike_time_i = jnp.max(
                    res["lif"].spike.time[0, res["lif"].spike.idx[0, :] == i]
                )

                # last output spike before first input spike -> no gradient
                if spike_time_i < spike_time_j:
                    self.assertEqual(gradient, 0.0)
                    continue

                # Simultaneous input/output spikes are a degenerate point of
                # the eventprop gradient formula, and are also zero.
                if spike_time_i == spike_time_j:
                    self.assertEqual(gradient, 0.0)
                    continue

                self.assertNotEqual(gradient, 0.0)


if __name__ == '__main__':
    unittest.main()
