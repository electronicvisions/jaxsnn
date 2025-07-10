import jax.numpy as jnp
from jaxsnn.event.loss import first_spike, nll_loss, target_time_loss
from jaxsnn.event.types import Spike
from numpy.testing import assert_almost_equal, assert_array_equal
import unittest


class TestEventLoss(unittest.TestCase):
    def test_first_spikes(self):
        spikes = Spike(jnp.array([0.1, 0.3]), jnp.array([-1, -1]))
        self.assertIsNone(
            assert_array_equal(
                first_spike(spikes, 2, 2), jnp.array([jnp.inf, jnp.inf])))

        spikes = Spike(jnp.array([0.1, 0.3]), jnp.array([1, -1]))
        self.assertIsNone(
            assert_array_equal(
                first_spike(spikes, 2, 2), jnp.array([jnp.inf, 0.1])))

        spikes = Spike(jnp.array([0.3, 0.1]), jnp.array([0, 0]))
        self.assertIsNone(
            assert_array_equal(
                first_spike(spikes, 2, 2), jnp.array([0.1, jnp.inf])))

    def test_custom_exp_loss(self):
        t_max = 3e-2
        target = jnp.array([0.5, 0.2]) * t_max

        t_spike = jnp.array([1.0, 1.0]) * t_max
        self.assertIsNone(
            assert_almost_equal(
                target_time_loss(t_spike, target, t_max), -0.845, 3))

        t_spike = jnp.array([0.5, jnp.inf]) * t_max
        self.assertIsNone(
            assert_almost_equal(
                target_time_loss(t_spike, target, t_max), -0.693, 3))

        t_spike = jnp.array([jnp.inf, jnp.inf]) * t_max
        self.assertEqual(target_time_loss(t_spike, target, t_max), 0.0)
        self.assertEqual(
            target_time_loss(target, target, t_max), -2 * jnp.log(2))

    def test_nll_loss(self):
        target = jnp.array([0, 1])
        self.assertIsNone(
            assert_almost_equal(
                nll_loss(jnp.array([0.0, 1.0]), target), 1.313, 3))
        self.assertIsNone(
            assert_almost_equal(
                nll_loss(jnp.array([1.0, 1.0]), target), 0.693, 3))
        self.assertIsNone(
            assert_almost_equal(
                nll_loss(jnp.array([1.0, 0.0]), target), 0.313, 3))


if __name__ == '__main__':
    unittest.main()
