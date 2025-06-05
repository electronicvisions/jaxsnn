import unittest

from pathlib import Path

import jax
import jax.numpy as jnp
from jaxsnn.event.modules.lif import LIF
from jaxsnn.event.modules.lif.parameters import LIFParameters
from jaxsnn.event.types import Spike, Step
from jaxsnn.event.states import LIFState


class TestLIF(unittest.TestCase):

    plot_path = Path(__file__).parent.joinpath("plots")

    def setUp(self):
        self.plot_path.mkdir(exist_ok=True)

    def test_lif_population_creation(self):
        """Test that LIF creates a Population with correct parameters."""
        params = LIFParameters(
            v_reset=0.0,
            v_th=1.0,
            tau_mem=20e-3,
            tau_syn=5e-3,
        )
        size = 3
        n_steps = 10

        population = LIF(size=size, n_steps=n_steps, params=params)

        # Check that population has correct size and parameters
        self.assertEqual(population.size, size)
        self.assertEqual(population.parameters["size"], size)
        self.assertEqual(population.parameters["n_steps"], n_steps)
        self.assertEqual(population.parameters["lif_params"], params)

    def test_lif_generator_functions(self):
        """
        Test that the generator creates proper init, state, and event
        functions
        """
        params = LIFParameters(
            v_reset=0.0,
            v_th=1.0,
            tau_mem=20e-3,
            tau_syn=5e-3,
        )
        size = 3
        n_steps = 10
        t_max = 100e-3

        population = LIF(size=size, n_steps=n_steps, params=params)

        # Generate functions with minimal topology
        pre_layer_pop_nodes = ["input"]
        pre_layer_param_nodes = ["syn"]
        pre_layer_params = {"syn": jnp.array([[0.]])}
        node_index_mapping = {"input": 0, "nrn": 1}
        node = "nrn"
        scc_mask = []
        backprop_method = "analytical"

        functions = population.generator(
            pre_layer_pop_nodes=pre_layer_pop_nodes,
            pre_layer_param_nodes=pre_layer_param_nodes,
            pre_layer_params=pre_layer_params,
            node_index_mapping=node_index_mapping,
            t_max=t_max,
            node=node,
            scc_mask=scc_mask,
            backprop_method=backprop_method,
        )

        # Test init function
        rng = jax.random.PRNGKey(0)
        init_result = functions.init(rng)
        self.assertIsNone(init_result)

        # Test state function
        state = functions.state()
        self.assertIsInstance(state.neuron_state, LIFState)
        self.assertEqual(state.neuron_state.V.shape, (size,))
        self.assertEqual(state.neuron_state.I.shape, (size,))

        # Test event function
        spike = functions.event(n_steps)
        self.assertIsInstance(spike, Spike)
        self.assertEqual(spike.time.shape, (n_steps,))
        self.assertEqual(spike.idx.shape, (n_steps,))

    def test_lif_step_function(self):
        """Test that the step function produces correct spike outputs."""
        params = LIFParameters(
            v_reset=0.0,
            v_th=1.0,
            tau_mem=20e-3,
            tau_syn=5e-3,
        )
        size = 2
        n_steps = 5
        t_max = 100e-3

        population = LIF(size=size, n_steps=n_steps, params=params)

        # Generate step function
        pre_layer_pop_nodes = ["input"]
        pre_layer_param_nodes = ["syn"]
        node_index_mapping = {"input": 0, "nrn": 1}
        pre_layer_params = {"input": 0.0}
        
        functions = population.generator(
            pre_layer_pop_nodes=pre_layer_pop_nodes,
            pre_layer_param_nodes=pre_layer_param_nodes,
            pre_layer_params=pre_layer_params,
            node_index_mapping=node_index_mapping,
            t_max=t_max,
            node="nrn",
            scc_mask=[],
            backprop_method="analytical",
        )

        # Create initial state
        state = functions.state()

        # Create input spikes
        input_spike = Spike.empty(n_steps)
        input_spike.time = input_spike.time.at[0].set(10e-3)

        spikes_dict = {"input": input_spike}
        external_spikes_dict = {}

        # Create weights
        weights = {"syn": jnp.array([[5.0], [3.0]])}

        # Create queue structures
        queue_heads = jnp.array([0])
        queue_indices = jnp.zeros(n_steps, dtype=int)

        # Test step function
        step_input = Step(
            parameters=weights,
            spikes=spikes_dict,
            external_spikes=external_spikes_dict,
            state=state,
            step_idx=0,
            layer_idx=1,
            queue_head=queue_heads,
            queue_indices=queue_indices,
        )

        spike_out, state_out, _, _ = functions.step(
            step_input
        )

        # Check that output has correct shape
        self.assertEqual(spike_out.time.shape, ())
        self.assertEqual(spike_out.idx.shape, ())
        self.assertIsInstance(state_out.neuron_state, LIFState)

    def test_lif_with_eventprop(self):
        """
        Test that eventprop backprop method creates adjoint step function.
        """
        params = LIFParameters(
            v_reset=0.0,
            v_th=1.0,
            tau_mem=20e-3,
            tau_syn=5e-3,
        )
        size = 3
        n_steps = 10

        population = LIF(size=size, n_steps=n_steps, params=params)

        # Generate with eventprop
        functions = population.generator(
            pre_layer_pop_nodes=["input"],
            pre_layer_param_nodes=["weight_input"],
            pre_layer_params={"weight_input": 0.0},
            node_index_mapping={"input": 0, "layer1": 1},
            t_max=100e-3,
            node="layer1",
            scc_mask=[],
            backprop_method="eventprop",
        )

        # Check that adjoint step function exists
        self.assertIsNotNone(functions.adjoint_step)


if __name__ == "__main__":
    unittest.main()
