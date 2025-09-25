import unittest
import jax
import jax.numpy as jnp
import numpy as np
from jaxsnn.base.types import LIFState
from jaxsnn.event.types import EventPropSpike
from jaxsnn.event.loss import (
    max_over_time,
    nll_loss,
    target_time_loss,
    ttfs_loss,
    mse_loss,
    first_spike,
)
from numpy.testing import assert_almost_equal, assert_array_equal


class TestLossFunctions(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        self.rng = jax.random.PRNGKey(42)
        self.tau_mem = 20.0

    def test_max_over_time(self):
        """Test max_over_time function with LIFState."""
        # Create test LIFState with time-varying voltage
        V = jnp.array([[0.5, 0.8, 0.3], [0.7, 0.2, 0.9], [0.1, 0.6, 0.4]])  # [time, neurons]
        I = jnp.zeros_like(V)
        lif_state = LIFState(V=V, I=I)

        result = max_over_time(lif_state)
        expected = jnp.array([0.7, 0.8, 0.9])  # max over time axis

        # Use numpy testing for array comparison
        np.testing.assert_array_almost_equal(result, expected, decimal=6)
        self.assertEqual(result.shape, (3,))

    def test_nll_loss(self):
        """Test negative log-likelihood loss function."""
        # Test case 1: Simple 3-class classification
        output = jnp.array([2.0, 1.0, 3.0])
        targets = jnp.array([0.0, 0.0, 1.0])  # class 2 is target (minimum value)

        loss = nll_loss(output, targets)

        # Should be finite and positive
        self.assertTrue(jnp.isfinite(loss))
        self.assertGreater(loss, 0.0)

        # Test case 2: Perfect prediction (target class has highest output)
        output_perfect = jnp.array([1.0, 1.0, 4.0])
        targets_perfect = jnp.array([1.0, 1.0, 0.0])  # class 2 is target

        loss_perfect = nll_loss(output_perfect, targets_perfect)
        self.assertTrue(jnp.isfinite(loss_perfect))
        self.assertLess(loss_perfect, loss)  # Perfect prediction should have lower loss

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

    def test_target_time_loss(self):
        """Test target time loss function."""
        first_spikes = jnp.array([10.0, 15.0, 20.0])
        target = jnp.array([12.0, 14.0, 18.0])

        loss = target_time_loss(first_spikes, target, self.tau_mem)

        # Should be finite and negative (log of positive values)
        self.assertTrue(jnp.isfinite(loss))
        self.assertLess(loss, 0.0)

        # Test perfect match
        loss_perfect = target_time_loss(target, target, self.tau_mem)
        self.assertLess(loss_perfect, loss)  # Perfect match should have lower (more negative) loss

        # Additional test cases from original test_loss.py
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

    def test_ttfs_loss(self):
        """Test time-to-first-spike (TTFS) loss function."""
        first_spikes = jnp.array([10.0, 15.0, 25.0])
        target = jnp.array([12.0, 8.0, 20.0])  # class 1 is target (minimum)

        loss = ttfs_loss(first_spikes, target, self.tau_mem)

        # Should be finite and negative
        self.assertTrue(jnp.isfinite(loss))
        self.assertLess(loss, 0.0)

        # Test case where target neuron (class 1) has better relative
        # performance
        first_spikes_better = jnp.array([25.0, 8.0, 30.0])  # class 1 spikes much earlier than others
        target_better = jnp.array([12.0, 8.0, 20.0])  # class 1 is still target

        loss_better = ttfs_loss(
            first_spikes_better, target_better, self.tau_mem)
        self.assertTrue(jnp.isfinite(loss_better))
        # Better relative performance should have higher (less negative) loss value
        self.assertGreater(loss_better, loss)

    def test_mse_loss(self):
        """Test mean squared error loss function."""
        first_spikes = jnp.array([10.0, 15.0, 20.0])
        target = jnp.array([12.0, 14.0, 18.0])

        loss = mse_loss(first_spikes, target, self.tau_mem)

        # Should be finite and non-negative
        self.assertTrue(jnp.isfinite(loss))
        self.assertGreaterEqual(loss, 0.0)

        # Test perfect match
        loss_perfect = mse_loss(target, target, self.tau_mem)
        np.testing.assert_array_almost_equal(loss_perfect, 0.0, decimal=6)

        # Test that loss increases with larger errors
        target_worse = jnp.array([5.0, 25.0, 35.0])  # Larger errors
        loss_worse = mse_loss(first_spikes, target_worse, self.tau_mem)
        self.assertGreater(loss_worse, loss)

    def test_first_spike(self):
        """Test first spike extraction from EventPropSpike."""
        n_outputs = 3
        times = jnp.array([5.0, 10.0, 15.0, 8.0, 12.0, 20.0])
        idx = jnp.array([0, 1, 2, 1, 0, 2])  # Neuron indices
        currents = jnp.ones_like(times)

        spikes = EventPropSpike(
            time=times,
            idx=idx,
            current=currents,
        )

        result = first_spike(spikes, n_outputs, n_outputs)
        expected = jnp.array([5.0, 8.0, 15.0])  # First internal spike for each neuron

        # Use numpy testing for array comparison
        np.testing.assert_array_almost_equal(result, expected, decimal=6)
        self.assertEqual(result.shape, (n_outputs,))

        spikes = EventPropSpike(
            time=jnp.array([0.1, 0.3]),
            idx=2 + jnp.array([0, 1]),
            current=jnp.array([1.0, 1.0]),
        )
        self.assertIsNone(
            assert_array_equal(
                first_spike(spikes, 2, 2), jnp.array([jnp.inf, jnp.inf])))

        # Test case: one internal spike for neuron 1
        spikes = EventPropSpike(
            time=jnp.array([0.1, 0.3]),
            idx=jnp.array([1, 3]),
            current=jnp.array([1.0, 1.0]),
        )
        self.assertIsNone(
            assert_array_equal(
                first_spike(spikes, 2, 2), jnp.array([jnp.inf, 0.1])))

        # Test case: internal spikes for both neurons
        spikes = EventPropSpike(
            time=jnp.array([0.3, 0.1]),
            idx=jnp.array([3, 0]),
            current=jnp.array([1.0, 1.0]),
        )
        self.assertIsNone(
            assert_array_equal(
                first_spike(spikes, 2, 2), jnp.array([0.1, jnp.inf])))

    def test_first_spike_no_spikes(self):
        """Test first spike when some neurons don't spike."""
        n_outputs = 3
        times = jnp.array([5.0, 10.0])
        idx = jnp.array([0, 1])  # Only neurons 0 and 1 spike
        currents = jnp.ones_like(times)

        spikes = EventPropSpike(
            time=times,
            idx=idx,
            current=currents,
        )

        result = first_spike(spikes, n_outputs, n_outputs)
        expected = jnp.array([5.0, 10.0, jnp.inf])  # neuron 2 never spikes
        np.testing.assert_array_equal(result, expected)

    def test_first_spike_external_only(self):
        """Test first spike when all spikes are external (should be ignored)."""
        n_outputs = 2
        times = jnp.array([5.0, 10.0])
        idx = jnp.array([2, 3])
        currents = jnp.ones_like(times)

        spikes = EventPropSpike(
            time=times,
            idx=idx,
            current=currents,
        )

        result = first_spike(spikes, n_outputs, n_outputs)
        expected = jnp.array([jnp.inf, jnp.inf])  # No internal spikes
        np.testing.assert_array_equal(result, expected)

    def test_loss_functions_shapes(self):
        """Test that all loss functions return scalar values."""
        # Setup test data
        first_spikes = jnp.array([10.0, 15.0, 20.0])
        target = jnp.array([12.0, 14.0, 18.0])
        output = jnp.array([2.0, 1.0, 3.0])

        # Test all scalar-returning loss functions
        losses = [
            target_time_loss(first_spikes, target, self.tau_mem),
            ttfs_loss(first_spikes, target, self.tau_mem),
            mse_loss(first_spikes, target, self.tau_mem),
            nll_loss(output, target),
        ]

        for loss in losses:
            self.assertEqual(loss.shape, ())  # Scalar
            self.assertTrue(jnp.isfinite(loss))


if __name__ == '__main__':
    unittest.main()
