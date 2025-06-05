import unittest
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from jaxsnn.event.states import LIFState
from jaxsnn.event.types import (
    Spike,
    StepState,
)
from jaxsnn.event.stepping import (
    step,
    multi_layer_step,
    next_input,
    min_delay_check,
)
from jaxsnn.event.functional import trajectory
from jaxsnn.event.functional.lif.dynamics import lif_exponential_flow
from jaxsnn.event.functional.lif.transition import transition
from jaxsnn.event.solver.next_finder import next_event
from jaxsnn.event.solver import ttfs_solver


class TestStep(unittest.TestCase):

    input_size = 10
    layer_size = 5
    t_max = 10.0
    n_steps = 20
    n_inputs = 7
    n_outputs = 12

    def run_single_layer_scc(
        self,
        input_spikes,
        weight: float = 1.0,
    ):

        parameters = {
            "inp": None,
            "syn": jnp.full((self.input_size, self.layer_size), weight),
            "nrn": None,
        }

        output_spikes = Spike.empty(self.n_steps)
        spikes = {
            "inp": input_spikes,
            "syn": None,
            "nrn": output_spikes
        }

        states = {
            "inp": None,
            "syn": None,
            "nrn": StepState(
                LIFState(
                    jnp.zeros(self.layer_size),
                    jnp.zeros(self.layer_size),
                ),
                jnp.array(0.0)
            )
        }
        queue_heads = {
            "inp": jnp.array(2 * [0]),
            "syn": jnp.array(2 * [0]),
            "nrn": jnp.array(2 * [0])
        }

        # Solver for next event
        solver = partial(
            ttfs_solver,
            10.0,
            10.0,
            1.0,
        )
        solver_fn = partial(
            next_event,
            jax.vmap(solver, in_axes=(0, None))
        )

        transition_fns = [
            partial(transition, 0.0, "syn"),
        ]

        single_flow = lif_exponential_flow(
            jnp.array(10.),
            jnp.array(10.),
        )
        dynamics_fn = jax.vmap(
            single_flow,
            in_axes=(0, None),
        )

        next_input_fn = partial(
            next_input,
            ["inp"],
            {"inp": 0.0},
        )

        min_delay_check_fn = partial(
            min_delay_check,
            ["inp"],
            {"inp": 0.0},
        )

        step_fn = partial(
            step,
            next_input_fn,
            min_delay_check_fn,
            dynamics_fn,
            transition_fns,
            self.t_max,
            solver_fn,
        )

        node_index_mapping = {"nrn": 2}
        step_fn = partial(
            multi_layer_step,
            {"nrn": step_fn},
            ["nrn"],
            node_index_mapping,
        )

        spikes, states, _, queue_idx = trajectory(
            step_fn,
            self.n_steps,
            parameters,
            spikes,
            None,
            states,
            queue_heads,
        )

        return spikes, states, queue_idx

    def test_step_feedforward_no_inputs(self):
        input_spikes = Spike().empty((self.n_inputs,))

        spikes, states, queue_idx = self.run_single_layer_scc(
            input_spikes
        )
        inputs = spikes["inp"]
        spikes = spikes["nrn"]
        states = states["nrn"]

        # Input event should stay the same
        self.assertTrue(jnp.all(inputs.time == input_spikes.time))
        self.assertTrue(jnp.all(inputs.idx == input_spikes.idx))
        self.assertTrue(jnp.all(inputs.current == input_spikes.current))
        self.assertTrue(jnp.all(inputs.layer_idx == input_spikes.layer_idx))
        self.assertTrue(jnp.all(inputs.internal == input_spikes.internal))

        # We should get n_steps events
        self.assertEqual(spikes.shape_, (self.n_steps,))

        # All events should be empty (no spikes)
        self.assertTrue(jnp.all(spikes.time == self.t_max))
        self.assertTrue(jnp.all(spikes.idx == -1))
        self.assertTrue(jnp.all(spikes.current == 0.))
        self.assertTrue(jnp.all(spikes.layer_idx == -1))
        self.assertTrue(jnp.all(~spikes.internal))

        # States should be zero
        self.assertTrue(jnp.all(states.neuron_state.V == 0.))
        self.assertTrue(jnp.all(states.neuron_state.I == 0.))

    def test_step_feedforward_no_internal(self):
        rng = jax.random.PRNGKey(np.random.randint(0, 10000))
        input_spikes = Spike(
            time=jax.random.uniform(
                rng, shape=(self.n_inputs,), minval=0.0, maxval=self.t_max,
            ),
            idx=jax.random.randint(
                rng, shape=(self.n_inputs,), minval=0,
                maxval=self.input_size,
            ),
            layer_idx=jnp.zeros(self.n_inputs, dtype=int),
            internal=jnp.ones(self.n_inputs, dtype=bool),
            current=jnp.zeros(self.n_inputs),
        )
        input_spikes = input_spikes.sort()

        spikes, states, queue_idx = self.run_single_layer_scc(
            input_spikes,
            weight=0.0,
        )
        inputs = spikes["inp"]
        spikes = spikes["nrn"]
        states = states["nrn"]

        # Input event should stay the same
        self.assertTrue(jnp.all(inputs.time == input_spikes.time))
        self.assertTrue(jnp.all(inputs.idx == input_spikes.idx))
        self.assertTrue(jnp.all(inputs.current == input_spikes.current))
        self.assertTrue(jnp.all(inputs.layer_idx == input_spikes.layer_idx))
        self.assertTrue(jnp.all(inputs.internal == input_spikes.internal))

        # Neuron spikes should hold input spikes
        nrn_input_spikes = spikes[:self.n_inputs]
        self.assertTrue(jnp.all(nrn_input_spikes.time == input_spikes.time))
        self.assertTrue(jnp.all(nrn_input_spikes.idx == input_spikes.idx))
        self.assertTrue(
            jnp.all(nrn_input_spikes.current == input_spikes.current)
        )
        self.assertTrue(
            jnp.all(nrn_input_spikes.layer_idx == input_spikes.layer_idx)
        )
        self.assertTrue(
            jnp.all(~nrn_input_spikes.internal == input_spikes.internal)
        )

        # We should get n_steps events
        self.assertEqual(spikes.shape_, (self.n_steps,))

        # # All internal events should be empty (no spikes)
        spikes = spikes.get_internal()
        self.assertEqual(spikes.shape_, (20,))
        self.assertTrue(jnp.all(~spikes.internal))

        # States should be zero
        self.assertTrue(jnp.all(states.neuron_state.V == 0.))
        self.assertTrue(jnp.all(states.neuron_state.I == 0.))

    def test_step_feedforward_realistic(self):
        rng = jax.random.PRNGKey(np.random.randint(0, 10000))
        input_spikes = Spike(
            time=jax.random.uniform(
                rng, shape=(self.n_inputs,), minval=0.0, maxval=self.t_max,
            ),
            idx=jax.random.randint(
                rng, shape=(self.n_inputs,), minval=0,
                maxval=self.input_size,
            ),
            layer_idx=jnp.zeros(self.n_inputs, dtype=int),
            internal=jnp.ones(self.n_inputs, dtype=bool),
            current=jnp.zeros(self.n_inputs),
        )
        input_spikes = input_spikes.sort()

        spikes, states, queue_idx = self.run_single_layer_scc(
            input_spikes,
            weight=1.5,
        )
        inputs = spikes["inp"]
        spikes = spikes["nrn"]
        states = states["nrn"]

        # Input event should stay the same
        self.assertTrue(jnp.all(inputs.time == input_spikes.time))
        self.assertTrue(jnp.all(inputs.idx == input_spikes.idx))
        self.assertTrue(jnp.all(inputs.current == input_spikes.current))
        self.assertTrue(jnp.all(inputs.layer_idx == input_spikes.layer_idx))
        self.assertTrue(jnp.all(inputs.internal == input_spikes.internal))

        # Neuron spikes should hold input spikes
        nrn_input_spikes = spikes.where(
            spikes.layer_idx == 0
        ).sort()[:self.n_inputs]
        self.assertTrue(
            jnp.all(nrn_input_spikes.time == input_spikes.time)
        )
        self.assertTrue(
            jnp.all(nrn_input_spikes.idx == input_spikes.idx)
        )
        self.assertTrue(
            jnp.all(nrn_input_spikes.current == input_spikes.current)
        )
        self.assertTrue(
            jnp.all(nrn_input_spikes.layer_idx == input_spikes.layer_idx)
        )
        self.assertTrue(
            jnp.all(~nrn_input_spikes.internal == input_spikes.internal)
        )

        # We should get n_steps events
        self.assertEqual(spikes.shape_, (self.n_steps,))

        # Assert we get internal events
        spikes = spikes.get_internal()
        self.assertGreater(spikes.shape_[0], 0)

        # States should be zero
        self.assertTrue(jnp.all(states.neuron_state.V != 0.))
        self.assertTrue(jnp.all(states.neuron_state.I != 0.))

    @unittest.skip("TODO: Test recurrent model.")
    def test_step_recurrent(self):
        pass


if __name__ == '__main__':
    unittest.main()
