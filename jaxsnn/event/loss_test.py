import jax.numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal

from jaxsnn.base.types import Spike
from jaxsnn.event.loss import first_spike, log_loss


def test_first_spikes():
    spikes = Spike(np.array([0.1, 0.3]), np.array([-1, -1]))
    assert_array_equal(first_spike(spikes, 2), np.array([np.inf, np.inf]))

    spikes = Spike(np.array([0.1, 0.3]), np.array([1, -1]))
    assert_array_equal(first_spike(spikes, 2), np.array([np.inf, 0.1]))

    spikes = Spike(np.array([0.3, 0.1]), np.array([0, 0]))
    assert_array_equal(first_spike(spikes, 2), np.array([0.1, np.inf]))


def test_log_loss():
    t_max = 3e-2
    target = np.array([0.5, 0.2]) * t_max

    t_spike = np.array([1.0, 1.0]) * t_max
    assert_almost_equal(log_loss(t_spike, target, t_max), -0.845, 3)

    t_spike = np.array([0.5, np.inf]) * t_max
    assert_almost_equal(log_loss(t_spike, target, t_max), -0.693, 3)

    t_spike = np.array([np.inf, np.inf]) * t_max
    assert log_loss(t_spike, target, t_max) == 0.0

    assert log_loss(target, target, t_max) == -2 * np.log(2)
