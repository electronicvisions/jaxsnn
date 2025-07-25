from functools import partial

import jax
import jax.numpy as jnp
from jaxsnn.base.params import LIFParameters
from jaxsnn.event.types import LIFState
from jaxsnn.event.root import cr_newton_solver
import unittest


class TestEventRootCustomRootNewton(unittest.TestCase):
    def get_lif_dynamics(self):
        params = LIFParameters()
        kernel = jnp.array(
            [[-1. / params.tau_mem, 1. / params.tau_mem],
             [0, -1. / params.tau_syn]])

        def f(state, time):
            initial_state = jnp.array([state.V, state.I])
            return jnp.dot(jax.scipy.linalg.expm(kernel * time), initial_state)

        def jc(state, t):
            return f(state, t)[0] - params.v_th

        return jc

    def test_cr_newton_solver(self):
        solver = partial(cr_newton_solver, self.get_lif_dynamics(), 0.0)

        def loss(weight):
            state = LIFState(V=0.0, I=3.0)
            state.I = state.I * weight
            return solver(state, dt=0.2)

        weight = jnp.array(1.0)
        value, grad = jax.value_and_grad(loss)(weight)
        self.assertAlmostEqual(value, 0.00323507, 8)
        self.assertAlmostEqual(grad, -0.00618034, 8)

    def test_cr_newton_solver_no_spike(self):
        solver = partial(cr_newton_solver, self.get_lif_dynamics(), 0.0)
        dt = 0.2

        def loss(weight):
            state = LIFState(V=0.0, I=2.0)
            state.I = state.I * weight
            return solver(state, dt)

        weight = jnp.array(1.0)
        value, grad = jax.value_and_grad(loss)(weight)
        self.assertEqual(value, dt)
        self.assertEqual(grad, 0)


if __name__ == '__main__':
    unittest.main()
