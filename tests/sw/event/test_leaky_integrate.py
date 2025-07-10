import jax.numpy as jnp
from jaxsnn.base.params import LIParameters
from jaxsnn.event.modules.leaky_integrate import LeakyIntegrator, li_cell
from jaxsnn.event.types import Spike
from numpy.testing import assert_array_almost_equal, assert_array_equal
import unittest


class TestEventLi(unittest.TestCase):
    def test_li_cell(self):
        params = LIParameters()
        t_max = 3.0 * params.tau_syn
        time_steps = jnp.linspace(0, t_max, 100)
        n_input = 3
        kernel = jnp.array(
            [[-1. / params.tau_mem, 1. / params.tau_mem],
             [0, -1. / params.tau_syn]])

        spikes = Spike(time=jnp.array([0]), idx=jnp.array([0]))
        weights = jnp.ones(n_input)
        res = li_cell(kernel, time_steps, weights, spikes)
        self.assertGreater(jnp.max(res.I), 0.95)
        self.assertEqual(jnp.argmax(res.I), 1)
        self.assertEqual(jnp.argmax(res.V), 46)

    def test_leaky_integrator(self):
        params = LIParameters()
        t_max = 3.0 * params.tau_syn
        n_hidden = 4
        n_input = 3
        time_steps = 100

        # test input spike previous layer
        spikes = Spike(time=jnp.array([1e-4]), idx=jnp.array([-1]))

        _, apply_fn = LeakyIntegrator(
            n_hidden, t_max, params, time_steps=time_steps
        )
        weights = jnp.arange(n_input * n_hidden).reshape(n_input, -1)
        res = apply_fn(weights, spikes)
        self.assertIsNone(
            assert_array_equal(res.I, jnp.zeros((time_steps, n_hidden))))
        self.assertIsNone(
            assert_array_equal(res.V, jnp.zeros((time_steps, n_hidden))))

        # test input spike
        spikes = Spike(
            time=jnp.array([0]),
            idx=jnp.array([0]),
        )
        _, apply_fn = LeakyIntegrator(n_hidden, t_max, params)
        res = apply_fn(weights, spikes)
        self.assertIsNone(
            assert_array_almost_equal(
                jnp.max(res.I, axis=0), [0.0, 0.85394, 1.707879, 2.561819]))
        self.assertIsNone(
            assert_array_almost_equal(
                jnp.max(res.V, axis=0),
                jnp.array([0.0, 0.249926, 0.499852, 0.749777])))

        # test inf spike time
        spikes = Spike(time=jnp.array([jnp.inf]), idx=jnp.array([-1]),)
        _, apply_fn = LeakyIntegrator(n_hidden, t_max, params)
        res = apply_fn(weights, spikes)

        self.assertFalse(jnp.any(jnp.isnan(res.V)))
        self.assertFalse(jnp.any(jnp.isnan(res.I)))


if __name__ == '__main__':
    unittest.main()
