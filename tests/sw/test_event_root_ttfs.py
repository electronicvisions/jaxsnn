import unittest
import numpy as np

import jax
import jax.numpy as jnp

from jaxsnn.event.modules.lif import LIFParameters
from jaxsnn.event.solver.lif_analytical import ttfs_solver
from jaxsnn.event.states import LIFState


params = LIFParameters()
t_max = 0.2


class TestEventRootTTFS(unittest.TestCase):

    def test_vmapping(self):
        expected_time = 0.003235072
        # Single neuron
        population_size = 10
        solver = ttfs_solver(
            params.tau_mem,
            params.tau_syn,
            params.v_th,
            params.v_leak,
        )
        state = LIFState(V=0.0, I=3.0)
        event = solver(state, t_max)

        self.assertEqual(event.shape, ())
        self.assertAlmostEqual(event, expected_time, 6)

        # Map over a population
        population_size = 10
        solver = ttfs_solver(
            params.tau_mem,
            params.tau_syn,
            params.v_th,
            params.v_leak,
        )
        pop_solver = jax.vmap(solver, in_axes=(0, None))
        state = LIFState(
            V=jnp.zeros(population_size), I=3.0 * jnp.ones(population_size))
        pop_events = pop_solver(state, t_max)

        self.assertEqual(pop_events.shape, (population_size,))
        self.assertIsNone(
            np.testing.assert_array_almost_equal(
                pop_events, jnp.full(population_size, expected_time)))

        # Map over batch
        batch_size = 32
        batched_pop_solver = jax.vmap(pop_solver, in_axes=(0, None))
        state = LIFState(
            V=jnp.zeros((batch_size, population_size)),
            I=3.0 * jnp.ones((batch_size, population_size)))
        batch_pop_events = batched_pop_solver(state, t_max)

        self.assertEqual(
            batch_pop_events.shape, (batch_size, population_size))
        self.assertIsNone(
            np.testing.assert_array_almost_equal(
                batch_pop_events,
                jnp.full((batch_size, population_size), expected_time)))


class TestEventRootTTFSGrads(unittest.TestCase):
    def test_ttfs_solver_vanishing_denomniator(self):
        solver = ttfs_solver(
            params.tau_mem,
            params.tau_mem / 2,
            params.v_th,
            params.v_leak,
        )
        def loss(weight):
            state = LIFState(V=-551.6683959960938, I=0.0006204545497894287)
            state.V = state.V * weight
            return solver(state, t_max)

        weight = jnp.array(1.0)
        value, grad = jax.value_and_grad(loss)(weight)
        self.assertEqual(value, t_max)
        self.assertEqual(grad, 0)

    def test_ttfs_solver_no_spike(self):
        # case tau_mem = 2*tau_syn
        solver = ttfs_solver(
            params.tau_mem,
            params.tau_mem / 2,
            params.v_th,
            params.v_leak,
        )
        def loss(weight):
            state = LIFState(V=0.0, I=2.0)
            state.I = state.I * weight
            return solver(state, t_max)

        weight = jnp.array(1.0)
        value, grad = jax.value_and_grad(loss)(weight)
        self.assertEqual(value, t_max)
        self.assertEqual(grad, 0)

        # case tau_mem = tau_syn
        solver = ttfs_solver(
            params.tau_syn,
            params.tau_syn,
            params.v_th,
            params.v_leak,
        )
        def loss(weight):
            state = LIFState(V=0.0, I=1.0)
            state.I = state.I * weight
            return solver(state, t_max)

        value, grad = jax.value_and_grad(loss)(weight)
        self.assertEqual(value, t_max)
        self.assertEqual(grad, 0)

    def test_ttfs_solver_spike(self):
        # test tau_mem = 2 * tau_syn
        solver = ttfs_solver(
            params.tau_mem,
            params.tau_mem / 2,
            params.v_th,
            params.v_leak,
        )
        def loss(weight):
            state = LIFState(V=0.0, I=3.0)
            state.I = state.I * weight
            return solver(state, t_max)

        weight = jnp.array(1.0)
        value, grad = jax.value_and_grad(loss)(weight)
        self.assertAlmostEqual(value, 0.00323507, 8)
        self.assertAlmostEqual(grad, -0.00618034, 8)

        # test tau_mem = tau_syn
        solver = ttfs_solver(
            params.tau_syn,
            params.tau_syn,
            params.v_th,
            params.v_leak,
        )
        def loss(weight):
            state = LIFState(V=0.0, I=3.0)
            state.I = state.I * weight
            return solver(state, t_max)

        value, grad = jax.value_and_grad(loss)(weight)
        self.assertAlmostEqual(value, 0.00129586, 8)
        self.assertAlmostEqual(grad, -0.0017492, 8)

    def test_nan(self):
        t_max = 4.0 * params.tau_syn
        neuron_state = LIFState(
            V=jnp.zeros(60),
            I=jnp.array(
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

        solver = ttfs_solver(
            params.tau_mem,
            params.tau_syn,
            params.v_th,
            params.v_leak,
        )
        batched_solver = jax.jit(jax.vmap(solver, in_axes=(0, None)))

        def loss_fn(weight):
            neuron_state.I = neuron_state.I * weight
            times = batched_solver(neuron_state, t_max)
            return jnp.sum(times)

        value, grad = jax.value_and_grad(loss_fn)(jnp.array(1.0))
        self.assertAlmostEqual(value, 1.04246, 4)
        self.assertAlmostEqual(grad, -0.12974, 4)

    def test_leak_over_threshold(self):
        params = LIFParameters(
            tau_mem=0.02,
            tau_syn=0.01,
            v_th=1.0,
            v_leak=2.0,
        )

        neuron_state = LIFState(
            V=0.0,
            I=0.0,
        )

        # no input current
        solver = ttfs_solver(
            params.tau_mem,
            params.tau_syn,
            params.v_th,
            params.v_leak,
        )
        time = solver(
            neuron_state,
            t_max,
        )
        self.assertAlmostEqual(time, 0.01386294, 7)

        # no input, voltage != 0
        neuron_state = LIFState(
            V=0.3,
            I=0.0,
        )
        # no input current
        solver = ttfs_solver(
            params.tau_mem,
            params.tau_syn,
            params.v_th,
            params.v_leak,
        )
        time = solver(
            neuron_state,
            t_max,
        )
        self.assertAlmostEqual(time, 0.01061257, 7)

        # with input current
        neuron_state = LIFState(
            V=0.3,
            I=0.001,
        )
        # no input current
        solver = ttfs_solver(
            params.tau_mem,
            params.tau_syn,
            params.v_th,
            params.v_leak,
        )
        time = solver(
            neuron_state,
            t_max,
        )
        self.assertAlmostEqual(time, 0.01061257, 5)


if __name__ == '__main__':
    unittest.main()
