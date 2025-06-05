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
    step_existing,
    multi_layer_step,
)
from jaxsnn.event.functional import trajectory
from jaxsnn.event.functional.lif.dynamics import lif_exponential_flow
from jaxsnn.event.functional.lif.transition import transition


class TestStepExistingEvents(unittest.TestCase):

    input_size = 10
    layer_size = 5
    t_max = 10.0
    n_steps = 35
    n_inputs = 7
    n_outputs = 12

    def run_single_layer_scc(
        self,
        input_spikes,
        external_spikes,
        weight: float = 1.0,
        recurrent: bool = False,
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
            "nrn": output_spikes,
        }
        external_spikes = {
            "nrn": external_spikes
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
            "nrn": jnp.array(2 * [0]) if not recurrent else jnp.array(3 * [0]),
        }

        if recurrent:
            parameters["syn_rec"] = jnp.ones(
                (self.layer_size, self.layer_size)
            )
            spikes["syn_rec"] = None
            states["syn_rec"] = None
            queue_heads["syn_rec"] = jnp.array(2 * [0])

        # Transitions per input connection
        if recurrent:
            transition_fns = [
                partial(transition, 0.0, "syn"),
                lambda s, w, i, l: s,
                partial(transition, 0.0, "syn_rec"),
            ]
        else:
            transition_fns = [
                partial(transition, 0.0, "syn"),
                lambda s, w, i, l: s,
                lambda s, w, i, l: s,
            ]
        single_flow = lif_exponential_flow(
            jnp.array(5),
            jnp.array(10),
        )
        dynamics_fn = jax.vmap(single_flow, in_axes=(0, None))

        step_fn = partial(
            step_existing,
            ["inp"] if not recurrent else ["inp", "nrn"],
            dynamics_fn,
            transition_fns,
            "nrn",
            self.t_max,
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
            external_spikes,
            states,
            queue_heads,
        )

        return spikes, states, queue_idx

    def generate_events(self):
        rng = jax.random.PRNGKey(np.random.randint(0, 10000))
        rng_input, rng_ext = jax.random.split(rng, 2)

        input_spikes = Spike(
            time=jax.random.uniform(
                rng_input, shape=(self.n_inputs,), minval=0.0, maxval=self.t_max,
            ),
            idx=jax.random.randint(
                rng_input, shape=(self.n_inputs,), minval=0,
                maxval=self.input_size,
            ),
            layer_idx=jnp.zeros(self.n_inputs, dtype=int),
            internal=jnp.ones(self.n_inputs, dtype=bool), 
            current=jnp.zeros(self.n_inputs),
        ).sort()

        external_spikes = Spike(
            time=jax.random.uniform(
                rng_ext, shape=(self.n_outputs,), minval=0.0, maxval=self.t_max,
            ),
            idx=jax.random.randint(
                rng_ext, shape=(self.n_outputs,), minval=0,
                maxval=self.layer_size,
            ),
            layer_idx=jnp.full(self.n_outputs, 2, dtype=int),
            internal=jnp.ones(self.n_outputs, dtype=bool),
            current=jnp.zeros(self.n_outputs),
        ).sort()

        return input_spikes, external_spikes, None

    def test_step_existing_feedforward_no_external(self):
        input_spikes, _, _ = self.generate_events()
        external_spikes = Spike.empty(self.n_steps)

        spikes, states, _ = self.run_single_layer_scc(
            input_spikes,
            external_spikes,
        )
        inputs = spikes["inp"]
        spikes = spikes["nrn"]
        states = states["nrn"]

        # We should get n_steps spikes
        self.assertTrue(spikes.shape_ == (self.n_steps,))

        # Input events should stay the same
        self.assertTrue(jnp.all(inputs.time == input_spikes.time))
        self.assertTrue(jnp.all(inputs.idx == input_spikes.idx))
        self.assertTrue(jnp.all(inputs.current == input_spikes.current))
        self.assertTrue(jnp.all(inputs.layer_idx == input_spikes.layer_idx))
        self.assertTrue(jnp.all(inputs.internal == input_spikes.internal))

        # Neuron spikes should ONLY hold input spikes
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

        if self.n_steps > self.n_inputs:
            nrn_input_spikes = spikes[self.n_inputs:]
            empty_spikes = Spike.empty(self.n_steps - self.n_inputs)

            self.assertTrue(jnp.all(nrn_input_spikes.time == self.t_max))
            self.assertTrue(jnp.all(nrn_input_spikes.idx == empty_spikes.idx))
            self.assertTrue(
                jnp.all(nrn_input_spikes.current == empty_spikes.current)
            )
            self.assertTrue(
                jnp.all(nrn_input_spikes.layer_idx == empty_spikes.layer_idx)
            )
            self.assertTrue(
                jnp.all(nrn_input_spikes.internal == empty_spikes.internal)
            )

        # States should not be zero (input events)
        self.assertTrue(jnp.any(states.neuron_state.V != 0.))
        self.assertTrue(jnp.any(states.neuron_state.I != 0.))

    def test_step_existing_feedforward(self):
        # Test random
        input_spikes, external_spikes, _ = self.generate_events()

        spikes, states, _ = self.run_single_layer_scc(
            input_spikes,
            external_spikes,
        )
        inputs = spikes["inp"]
        spikes = spikes["nrn"]
        states = states["nrn"]

        # We should get n_steps spikes
        self.assertTrue(spikes.shape_ == (self.n_steps,))

        # Input events should stay the same
        self.assertTrue(jnp.all(inputs.time == input_spikes.time))
        self.assertTrue(jnp.all(inputs.idx == input_spikes.idx))
        self.assertTrue(jnp.all(inputs.current == input_spikes.current))
        self.assertTrue(jnp.all(inputs.layer_idx == input_spikes.layer_idx))
        self.assertTrue(jnp.all(inputs.internal == input_spikes.internal))

        # Neuron spikes should hold input spikes and external spikes
        merged_spikes = input_spikes.concatenate(external_spikes)
        merged_spikes = merged_spikes.concatenate(
            Spike.empty(self.n_steps - merged_spikes.shape_[0])
        ).sort()
        # Time is capped at t_max
        merged_spikes.time = jnp.where(
            merged_spikes.time >= self.t_max, self.t_max, merged_spikes.time
        )
        # We need to adjust internal
        merged_spikes.internal = jnp.where(
            merged_spikes.layer_idx == 0, False, merged_spikes.internal
        )
        merged_spikes = merged_spikes[:self.n_steps]

        # We dont have current in external events
        nrn_input_spikes = spikes[:self.n_steps]
        nrn_input_spikes.current = jnp.zeros_like(merged_spikes.current)

        self.assertTrue(jnp.all(nrn_input_spikes.time == merged_spikes.time))
        self.assertTrue(jnp.all(nrn_input_spikes.idx == merged_spikes.idx))
        self.assertTrue(
            jnp.all(nrn_input_spikes.current == merged_spikes.current)
        )
        self.assertTrue(
            jnp.all(nrn_input_spikes.layer_idx == merged_spikes.layer_idx)
        )
        self.assertTrue(
            jnp.all(nrn_input_spikes.internal == merged_spikes.internal)
        )

        # States should not be zero
        if inputs.time.min() < spikes.time.max():
            self.assertTrue(jnp.any(states.neuron_state.V != 0.))
            self.assertTrue(jnp.any(states.neuron_state.I != 0.))

    def test_step_existing_feedforward_equal_times(self):
        input_spikes, external_spikes, _ = self.generate_events()

        # Test multiple equal input times
        input_spikes.time = input_spikes.time.at[3].set(input_spikes.time[4])
        input_spikes = input_spikes.sort()
        # Force equal external spike times
        external_spikes.time = external_spikes.time.at[2].set(
            external_spikes.time[3]
        )
        external_spikes = external_spikes.sort()

        # force external spike to be equal to input spike -> external first
        external_spikes.time = external_spikes.time.at[5].set(
            input_spikes.time[5]
        )
        external_spikes = external_spikes.sort()

        # force external spike to be equal to input spike -> input first
        input_spikes.time = input_spikes.time.at[6].set(
            external_spikes.time[6]
        )
        input_spikes = input_spikes.sort()

        # run
        spikes, states, _ = self.run_single_layer_scc(
            input_spikes,
            external_spikes,
        )
        inputs = spikes["inp"]
        spikes = spikes["nrn"]
        states = states["nrn"]

        # Input events should stay the same
        self.assertTrue(jnp.all(inputs.time == input_spikes.time))
        self.assertTrue(jnp.all(inputs.idx == input_spikes.idx))
        self.assertTrue(jnp.all(inputs.current == input_spikes.current))
        self.assertTrue(jnp.all(inputs.layer_idx == input_spikes.layer_idx))
        self.assertTrue(jnp.all(inputs.internal == input_spikes.internal))

        # Neuron spikes should hold input spikes and external spikes
        merged_spikes = input_spikes.concatenate(external_spikes)
        merged_spikes = merged_spikes.concatenate(
            Spike.empty(self.n_steps - merged_spikes.shape_[0])
        ).sort()
        # Time is capped at t_max
        merged_spikes.time = jnp.where(
            merged_spikes.time >= self.t_max, self.t_max, merged_spikes.time
        )
        # We need to adjust internal
        merged_spikes.internal = jnp.where(
            merged_spikes.layer_idx == 0, False, merged_spikes.internal
        )
        merged_spikes = merged_spikes[:self.n_steps]


        # We dont have current in external events
        nrn_input_spikes = spikes[:self.n_steps]
        nrn_input_spikes.current = jnp.zeros_like(merged_spikes.current)

        self.assertTrue(jnp.all(nrn_input_spikes.time == merged_spikes.time))
        self.assertTrue(jnp.all(nrn_input_spikes.idx == merged_spikes.idx))
        self.assertTrue(
            jnp.all(nrn_input_spikes.current == merged_spikes.current)
        )
        self.assertTrue(
            jnp.all(nrn_input_spikes.layer_idx == merged_spikes.layer_idx)
        )
        self.assertTrue(
            jnp.all(nrn_input_spikes.internal == merged_spikes.internal)
        )

        # States should not be zero
        if inputs.time.min() < spikes.time.max():
            self.assertTrue(jnp.any(states.neuron_state.V != 0.))
            self.assertTrue(jnp.any(states.neuron_state.I != 0.))

    def test_step_existing_recurrent(self):
        # Test random
        input_spikes, external_spikes, _ = self.generate_events()

        spikes, states, _ = self.run_single_layer_scc(
            input_spikes,
            external_spikes,
            recurrent=True,
        )
        inputs = spikes["inp"]
        spikes = spikes["nrn"]
        states = states["nrn"]

        # We should get n_steps spikes
        self.assertTrue(spikes.shape_ == (self.n_steps,))

        # Input events should stay the same
        self.assertTrue(jnp.all(inputs.time == input_spikes.time))
        self.assertTrue(jnp.all(inputs.idx == input_spikes.idx))
        self.assertTrue(jnp.all(inputs.current == input_spikes.current))
        self.assertTrue(jnp.all(inputs.layer_idx == input_spikes.layer_idx))
        self.assertTrue(jnp.all(inputs.internal == input_spikes.internal))

        # Neuron spikes should hold input spikes and rec external spikes
        merged_spikes = input_spikes.concatenate(external_spikes)
        external_spikes.internal = jnp.zeros_like(external_spikes.internal)
        # Add recurrent internal spieks
        merged_spikes = merged_spikes.concatenate(external_spikes).sort()
        merged_spikes = merged_spikes.concatenate(
            Spike.empty(self.n_steps - merged_spikes.shape_[0])
        ).sort()
        # Time is capped at t_max
        merged_spikes.time = jnp.where(
            merged_spikes.time >= self.t_max, self.t_max, merged_spikes.time
        )
        # We need to adjust internal
        merged_spikes.internal = jnp.where(
            merged_spikes.layer_idx == 0, False, merged_spikes.internal
        )
        merged_spikes = merged_spikes[:self.n_steps]

        # We dont have current in external events
        nrn_input_spikes = spikes[:self.n_steps]
        nrn_input_spikes.current = jnp.zeros_like(merged_spikes.current)

        self.assertTrue(jnp.all(nrn_input_spikes.time == merged_spikes.time))
        self.assertTrue(jnp.all(nrn_input_spikes.idx == merged_spikes.idx))
        self.assertTrue(
            jnp.all(nrn_input_spikes.current == merged_spikes.current)
        )
        self.assertTrue(
            jnp.all(nrn_input_spikes.layer_idx == merged_spikes.layer_idx)
        )
        self.assertTrue(
            jnp.all(nrn_input_spikes.internal == merged_spikes.internal)
        )

        # States should not be zero
        if inputs.time.min() < spikes.time.max():
            self.assertTrue(jnp.any(states.neuron_state.V != 0.))
            self.assertTrue(jnp.any(states.neuron_state.I != 0.))


    @unittest.skip("TODO: Test multiple input layers.")
    def test_step_existing_feedforward_multiple_inputs(self):
        pass


if __name__ == '__main__':
    unittest.main()
