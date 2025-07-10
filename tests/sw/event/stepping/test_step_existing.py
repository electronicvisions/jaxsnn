import unittest
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jaxsnn.event.utils.filter import filter_spikes
from jaxsnn.event.stepping import step_existing
from jaxsnn.event.types import (
    EventPropSpike, LIFState, StepState, WeightInput, InputQueue)
from jaxsnn.event.transition import transition_without_recurrence
from jaxsnn.event.flow import lif_exponential_flow
from jaxsnn.base.params import LIFParameters


class TestEventFlow(unittest.TestCase):

    def test_step_existing(self):

        layer_start = 10
        layer_size = 5
        weights = WeightInput(jnp.ones((layer_start, layer_size)))

        input_spikes = EventPropSpike(
            time=jnp.array([0.1, 0.3, 0.3, 0.6, 0.6, 0.2, 0.5, 0.4]),
            idx=jnp.array([2, 14, 13, 7, 12, 12, 1, 11]),
            current=jnp.zeros(8))

        input_spikes = filter_spikes(input_spikes, 0)

        step_state = StepState(
            neuron_state=LIFState(jnp.zeros(layer_size), jnp.zeros(layer_size)),
            spike_times=-1 * jnp.ones(layer_size),
            spike_mask=jnp.zeros(layer_size, dtype=bool),
            time=0.0,
            input_queue=InputQueue(input_spikes))

        params = LIFParameters()
        single_flow = lif_exponential_flow(params)
        dynamics = jax.vmap(single_flow, in_axes=(0, None))
        transition = partial(transition_without_recurrence, params)
        step_fn = partial(step_existing, dynamics, transition, 1.0, None)

        state, spikes = jax.lax.scan(
            step_fn, (step_state, weights, layer_start),
            jnp.arange(10))

        self.assertIsNone(
            np.testing.assert_array_equal(
                spikes.time, jnp.concatenate(
                    (input_spikes.time, jnp.array([jnp.inf, jnp.inf])))))
        self.assertIsNone(
            np.testing.assert_array_equal(
                spikes.idx, jnp.concatenate(
                    (input_spikes.idx, jnp.array([-1, -1])))))


if __name__ == '__main__':
    unittest.main()