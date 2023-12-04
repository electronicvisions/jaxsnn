import jax.numpy as np
from jaxsnn.event.flow import exponential_flow
from jaxsnn.event.leaky_integrate_and_fire import LIFState
from numpy.testing import assert_almost_equal, assert_array_almost_equal
import unittest


class TestEventFlow(unittest.TestCase):
    def test_exponential_flow(self):
        A = np.array([[-1, 1], [0, -1]])
        flow_function = exponential_flow(A)
        state = LIFState(V=1.0, I=1.0)
        new_state = flow_function(state, 1.0)
        assert_almost_equal(new_state.I, 0.368, 3)
        assert_almost_equal(new_state.V, 0.736, 3)

    def test_batched_exponential_flow(self):
        A = np.array([[-1, 1], [0, -1]])
        flow_function = exponential_flow(A)
        state = LIFState(V=np.full(10, 1.0), I=np.full(10, 1.0))
        new_state = flow_function(state, 1.0)
        assert_array_almost_equal(new_state.I, np.full(10, 0.368), 3)
        assert_array_almost_equal(new_state.V, np.full(10, 0.736), 3)


if __name__ == '__main__':
    unittest.main()
