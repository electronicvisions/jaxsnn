import unittest
import jax.numpy as jnp
from numpy.testing import assert_array_almost_equal
from jaxsnn.event.functional.lif.dynamics import lif_exponential_flow
from jaxsnn.event.states import LIFState


class TestEventFlow(unittest.TestCase):
    def test_exponential_flow(self):
        flow_function = lif_exponential_flow(tau_syn=1.0, tau_mem=1.0)
        state = LIFState(V=1.0, I=1.0)
        new_state = flow_function(state, 1.0)
        self.assertAlmostEqual(new_state.I, 0.368, 3)
        self.assertAlmostEqual(new_state.V, 0.736, 3)

    def test_batched_exponential_flow(self):
        flow_function = lif_exponential_flow(tau_syn=1.0, tau_mem=1.0)
        state = LIFState(V=jnp.full(10, 1.0), I=jnp.full(10, 1.0))
        new_state = flow_function(state, 1.0)
        self.assertIsNone(assert_array_almost_equal(
            new_state.I, jnp.full(10, 0.368), 3))
        self.assertIsNone(assert_array_almost_equal(
            new_state.V, jnp.full(10, 0.736), 3))


if __name__ == '__main__':
    unittest.main()
