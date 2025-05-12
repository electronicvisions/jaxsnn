from functools import partial
import numpy as np

import jax
import jax.numpy as jnp
from jaxsnn.base.params import LIFParameters
from jaxsnn.event.flow import exponential_flow
from jaxsnn.event.stepping import step
from jaxsnn.event.modules.leaky_integrate_and_fire import (
    transition_without_recurrence)
from jaxsnn.event.root.next_finder import next_event
from jaxsnn.event.root.ttfs import ttfs_solver
from jaxsnn.event.types import (
    LIFState,
    EventPropSpike,
    InputQueue,
    StepState,
    WeightInput)
import unittest


class TestStep(unittest.TestCase):
    def test_step_feedforward_transition(self):

        params = LIFParameters()
        kernel = jnp.array(
            [[-1. / params.tau_mem, 1. / params.tau_mem],
             [0, -1. / params.tau_syn]])
        flow = exponential_flow(kernel)
        dynamics = jax.vmap(flow, in_axes=(0, None))

        t_max = 10.0
        n_input = 2
        n_hidden = 2
        layer_start = 2
        weights = WeightInput(jnp.zeros((n_input, n_hidden)))

        spikes = EventPropSpike(
            time=jnp.array([1.0, 2.0]),
            idx=jnp.array([0, 1]),
            current=jnp.array([0.0, 0.0]))

        solver = partial(ttfs_solver, params.tau_mem, params.v_th)
        batched_solver = partial(next_event, jax.vmap(solver, in_axes=(0, None)))
        transition = partial(transition_without_recurrence, params)

        step_fn = partial(step, dynamics, transition, t_max, batched_solver)

        state = StepState(
            neuron_state=LIFState(jnp.zeros(n_hidden), np.zeros(n_hidden)),
            spike_times=-1 * jnp.ones(n_hidden),
            spike_mask=jnp.zeros(n_hidden, dtype=bool),
            time=0.0,
            input_queue=InputQueue(spikes))
        step_state = (state, weights, layer_start)

        step_state, spike = step_fn(step_state)
        self.assertEqual(spike.time, 1.0)
        self.assertEqual(spike.idx, 0)

        step_state, spike = step_fn(step_state)
        self.assertEqual(spike.time, 2.0)
        self.assertEqual(spike.idx, 1)

        step_state, spike = step_fn(step_state)
        self.assertEqual(spike.time, t_max)
        self.assertEqual(spike.idx, -1)

    def test_step_no_transition(self):
        params = LIFParameters()
        kernel = jnp.array(
            [[-1. / params.tau_mem, 1. / params.tau_mem],
             [0, -1. / params.tau_syn]])
        flow = exponential_flow(kernel)
        dynamics = jax.vmap(flow, in_axes=(0, None))

        t_max = 10.0
        n_input = 2
        n_hidden = 2
        layer_start = 2
        weights = WeightInput(jnp.ones((n_input, n_hidden)))

        spikes = EventPropSpike(
            time=jnp.array([1.0]),
            idx=jnp.array([0]),
            current=jnp.array([0.0]))

        solver = partial(ttfs_solver, params.tau_mem, params.v_th)
        batched_solver = partial(
            next_event, jax.vmap(solver, in_axes=(0, None)))
        transition = partial(transition_without_recurrence, params)

        step_fn = partial(step, dynamics, transition, t_max, batched_solver)

        # normal input spike, neuron current should increase
        state = StepState(
            neuron_state=LIFState(
                jnp.zeros(n_hidden), jnp.zeros(n_hidden)),
            spike_times=-1 * np.ones(n_hidden),
            spike_mask=jnp.zeros(n_hidden, dtype=bool),
            time=0.0,
            input_queue=InputQueue(spikes))
        step_state = (state, weights, layer_start)

        (step_state, weights, _), spike = step_fn(step_state)
        self.assertEqual(spike.time, 1.0)
        self.assertEqual(spike.idx, 0)
        self.assertEqual(step_state.input_queue.head, 1)
        self.assertIsNone(
            np.testing.assert_array_equal(
                step_state.neuron_state.I, np.ones(2)))

        # input spike of previous layer, neuron current should not increase
        state = StepState(
            neuron_state=LIFState(
                jnp.zeros(n_hidden), jnp.zeros(n_hidden)),
            spike_times=-1 * jnp.ones(n_hidden),
            spike_mask=jnp.zeros(n_hidden, dtype=bool),
            time=0.0,
            input_queue=InputQueue(spikes))
        layer_start = 5
        step_state = (state, weights, layer_start)

        (step_state, weights, _), spike = step_fn(step_state)
        self.assertEqual(spike.time, 1.0)
        self.assertEqual(spike.idx, 0)
        self.assertEqual(step_state.input_queue.head, 1)
        self.assertIsNone(
            np.testing.assert_array_equal(
                step_state.neuron_state.I, np.zeros(2)))


if __name__ == '__main__':
    unittest.main()
