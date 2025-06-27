from functools import partial

import jax
import jax.numpy as np
from jaxsnn.base.params import LIFParameters
from jaxsnn.event.modules.leaky_integrate import LIFState
from jaxsnn.event.root.ttfs import ttfs_solver
import unittest

params = LIFParameters()
t_max = 0.2


class TestEventRootTtfs(unittest.TestCase):
    def test_ttfs_solver_vanishing_denomniator(self):
        def loss(weight):
            state = LIFState(V=-551.6683959960938, I=0.0006204545497894287)
            state.V = state.V * weight
            return ttfs_solver(params.tau_mem, params.v_th, state, t_max)

        weight = np.array(1.0)
        value, grad = jax.value_and_grad(loss)(weight)
        self.assertEqual(value, t_max)
        self.assertEqual(grad, 0)

    def test_ttfs_solver_no_spike(self):
        def loss(weight):
            state = LIFState(V=0.0, I=2.0)
            state.I = state.I * weight
            return ttfs_solver(params.tau_mem, params.v_th, state, t_max)

        weight = np.array(1.0)
        value, grad = jax.value_and_grad(loss)(weight)
        self.assertEqual(value, t_max)
        self.assertEqual(grad, 0)

    def test_ttfs_solver_spike(self):
        def loss(weight):
            state = LIFState(V=0.0, I=3.0)
            state.I = state.I * weight
            return ttfs_solver(params.tau_mem, params.v_th, state, t_max)

        weight = np.array(1.0)
        value, grad = jax.value_and_grad(loss)(weight)
        self.assertAlmostEqual(value, 0.00323507, 8)
        self.assertAlmostEqual(grad, -0.00618034, 8)

    def test_nan(self):
        t_max = 4.0 * params.tau_syn
        neuron_state = LIFState(
            V=np.zeros(60),
            I=np.array(
                [
                    0.45193174,
                    3.19146,
                    0.7908741,
                    1.0539854,
                    0.56739396,
                    0.7030711,
                    1.3831071,
                    1.913287,
                    0.49313363,
                    2.508708,
                    2.335917,
                    0.42423773,
                    0.9986891,
                    0.6763091,
                    0.6967824,
                    1.8609539,
                    1.0740578,
                    1.8261349,
                    0.99030566,
                    2.1311684,
                    2.0386827,
                    1.0601723,
                    0.3659073,
                    1.2009021,
                    2.4924612,
                    2.7251928,
                    1.5452971,
                    1.6464705,
                    1.6182272,
                    2.121055,
                    2.5739164,
                    2.472322,
                    1.1995058,
                    1.9270785,
                    2.1202886,
                    1.451486,
                    0.53539944,
                    1.2206947,
                    2.8484209,
                    0.68368566,
                    0.43251452,
                    0.5766972,
                    2.637784,
                    1.8045906,
                    1.5158126,
                    2.9772224,
                    1.8381596,
                    2.6536105,
                    1.9222406,
                    1.2088894,
                    1.0864722,
                    1.0823696,
                    1.6205056,
                    2.3292239,
                    1.2918817,
                    0.657693,
                    1.7307178,
                    0.55521065,
                    2.3550713,
                    1.9288002,
                ]
            ),
        )

        solver = partial(ttfs_solver, params.tau_mem, params.v_th)
        batched_solver = jax.jit(jax.vmap(solver, in_axes=(0, None)))

        def loss_fn(weight):
            neuron_state.I = neuron_state.I * weight
            times = batched_solver(neuron_state, t_max)
            return np.sum(times)

        value, grad = jax.value_and_grad(loss_fn)(np.array(1.0))
        self.assertAlmostEqual(value, 1.04246, 4)
        self.assertAlmostEqual(grad, -0.12974, 4)


if __name__ == '__main__':
    unittest.main()
