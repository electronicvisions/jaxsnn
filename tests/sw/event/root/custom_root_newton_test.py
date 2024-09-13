from functools import partial

import jax
import jax.numpy as np
from jaxsnn.base.params import LIFParameters
from jaxsnn.event.types import LIFState
from jaxsnn.event.root import cr_newton_solver
from numpy.testing import assert_almost_equal
import unittest


class TestEventRootCustomRootNewton(unittest.TestCase):
    def get_lif_dynamics(self):
        params = LIFParameters()
        kernel = np.array(
            [[-1. / params.tau_mem, 1. / params.tau_mem],
             [0, -1. / params.tau_syn]])

        def f(state, time):
            initial_state = np.array([state.V, state.I])
            return np.dot(jax.scipy.linalg.expm(kernel * time), initial_state)

        def jc(state, t):
            return f(state, t)[0] - params.v_th

        return jc

    def test_cr_newton_solver(self):
        solver = partial(cr_newton_solver, self.get_lif_dynamics(), 0.0)

        def loss(weight):
            state = LIFState(V=0.0, I=3.0)
            state.I = state.I * weight
            return solver(state, dt=0.2)

        weight = np.array(1.0)
        value, grad = jax.value_and_grad(loss)(weight)
        assert_almost_equal(value, 0.00323507, 8)
        assert_almost_equal(grad, -0.00618034, 8)

    def test_cr_newton_solver_no_spike(self):
        solver = partial(cr_newton_solver, self.get_lif_dynamics(), 0.0)
        dt = 0.2

        def loss(weight):
            state = LIFState(V=0.0, I=2.0)
            state.I = state.I * weight
            return solver(state, dt)

        weight = np.array(1.0)
        value, grad = jax.value_and_grad(loss)(weight)
        assert value == dt
        assert grad == 0


if __name__ == '__main__':
    unittest.main()
