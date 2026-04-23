import unittest
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import numpy as np
from jaxsnn.event.modules.leaky_integrate_and_fire import EventPropLIF
from jaxsnn.event.utils.filter import filter_spikes
from jaxsnn.event.stepping import step_existing
from jaxsnn.event.types import (
    EventPropSpike, LIFState, StepState, WeightInput, InputQueue)
from jaxsnn.event.transition import transition_without_recurrence
from jaxsnn.event.flow import lif_exponential_flow
from jaxsnn.base.params import LIFParameters


class TestStepExistingEvents(unittest.TestCase):

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


class TestStepExistingEventsRealistic(unittest.TestCase):

    def test_step_existing_realistic_batched(self):

        rng = jax.random.PRNGKey(0)
        rng, rng_params = jax.random.split(rng)

        input_size = 100
        batch_size = 128
        hidden_size = 200
        n_input_spikes = 100
        n_spikes_hidden = 10 * hidden_size
        n_spikes_total = n_input_spikes + n_spikes_hidden
        t_max = 0.1
        params = LIFParameters()

        input_spikes = EventPropSpike(
            time=jax.random.uniform(
            rng, shape=(batch_size, n_input_spikes), minval=0.0, maxval=t_max),
            idx=jax.random.randint(
                rng, shape=(batch_size, n_input_spikes), minval=0,
                maxval=input_size),
            current=jnp.zeros((batch_size, n_input_spikes)))

        # Generate events
        init_fn, apply_fn = EventPropLIF(
            size=hidden_size,
            n_spikes=n_spikes_total,
            t_max=t_max,
            params=params,
            mean=0.2,
            std=0.1)
        apply_fn = jax.vmap(apply_fn, in_axes=(None, 0, 0, None))
        rng_params, _, weights = init_fn(rng_params, input_size)
        _, _, spikes, _ = apply_fn([weights], input_spikes, None, None)

        # Now step events
        single_flow = lif_exponential_flow(params)
        dynamics = jax.vmap(single_flow, in_axes=(0, None))
        transition = partial(transition_without_recurrence, params)
        step_fn = partial(step_existing, dynamics, transition, t_max, None)

        def step_fn_wrapped(times, idxs, currents):
            step_state = StepState(
                neuron_state=LIFState(
                    jnp.zeros(hidden_size),
                    jnp.zeros(hidden_size)),
                spike_times=-1 * jnp.ones(hidden_size),
                spike_mask=jnp.zeros(hidden_size, dtype=bool),
                time=0.0,
                input_queue=InputQueue(EventPropSpike(
                    time=times, idx=idxs, current=currents)))
            input_state = (step_state, weights, input_size)
            _, spikes = jax.lax.scan(
                step_fn, input_state, jnp.arange(n_spikes_total))
            return spikes

        step_fn_vmapped = jax.vmap(step_fn_wrapped, in_axes=(0, 0, 0))
        spikes = step_fn_vmapped(spikes.time, spikes.idx, spikes.current)

        self.assertEqual(spikes.time.shape, (batch_size, n_spikes_total))
        self.assertEqual(spikes.idx.shape, (batch_size, n_spikes_total))
        self.assertEqual(spikes.current.shape, (batch_size, n_spikes_total))


if __name__ == '__main__':
    unittest.main()