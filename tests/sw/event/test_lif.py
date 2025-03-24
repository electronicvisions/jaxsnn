import unittest

from functools import partial
import matplotlib.pyplot as plt
from pathlib import Path

import jax
import jax.numpy as np
import numpy as rnp
from jax import random
from jaxsnn.base.compose import serial
from jaxsnn.base.dataset.yinyang import yinyang_dataset
from jaxsnn.event.modules.leaky_integrate_and_fire import (
    LIF, EventPropLIF, LIFParameters)
from jaxsnn.event.types import EventPropSpike
from numpy.testing import assert_array_equal, assert_array_almost_equal
from jaxsnn.event.encode import (
    spatio_temporal_encode,
    target_temporal_encode,
    encode
)
from jaxsnn.event.types import WeightInput


class TestLIF(unittest.TestCase):

    plot_path = Path(__file__).parent.joinpath("plots")

    def setUp(self):
        self.plot_path.mkdir(exist_ok=True)

    def test_multiple_simultaneous_spikes(self):
        """ """
        # neuron weights, low v_reset only allows one spike per neuron
        params = LIFParameters(v_reset=-1_000.0, v_th=1.0)

        # declare net
        _, apply_fn = serial(
            LIF(
                3,
                n_spikes=4,
                t_max=100e-3,
                params=params,
                mean=0,
                std=0))

        # Same spike time through same input neuron
        weights = np.zeros((2, 3))
        weights = weights.at[0, 0].set(4.5)
        weights = weights.at[0, 2].set(4.5)
        weights = [WeightInput(weights)]

        inputs = EventPropSpike(
            np.array([10e-3]),
            np.array([0]),
            np.array([0.0]))

        res = apply_fn(weights, inputs)[-1]

        # Neuron 0 and 2 should spike at same time
        self.assertAlmostEqual(res[0].time[1], res[0].time[2], 5)
        self.assertNotEqual(res[0].time[1], 100e-3)

        # now check grads
        def loss_fn(apply_fn, weights, input_spikes):
            ret = apply_fn(weights, input_spikes)
            loss = ret[-1][0].time[2] - 0.5 * ret[-1][0].time[1]
            return loss

        loss_func = partial(loss_fn, apply_fn)

        # check gradients
        _, grad = jax.value_and_grad(loss_func)(weights, inputs)
        self.assertGreater(grad[0].input[0, 0], grad[0].input[0, 2])

        # Same spike time through different input neurons
        weights = np.zeros((2, 3))
        weights = weights.at[0, 0].set(4.5)
        weights = weights.at[1, 2].set(4.5)
        weights = [WeightInput(weights)]

        inputs = EventPropSpike(
            np.array([10e-3, 10e-3]),
            np.array([0, 1]),
            np.array([0.0, 0.0]))

        res = apply_fn(weights, inputs)[-1]

        # Neuron 0 and 2 should spike at same time
        self.assertAlmostEqual(res[0].time[2], res[0].time[3], 5)
        self.assertNotEqual(res[0].time[-1], 100e-3)

        def loss_fn(apply_fn, weights, input_spikes):
            ret = apply_fn(weights, input_spikes)
            loss = ret[-1][0].time[3] - 0.5 * ret[-1][0].time[2]
            return loss

        loss_func = partial(loss_fn, apply_fn)

        # check gradients
        _, grad = jax.value_and_grad(loss_func)(weights, inputs)
        self.assertAlmostEqual(grad[0].input[0, 0], grad[0].input[1, 0], 6)
        self.assertAlmostEqual(grad[0].input[0, 2], grad[0].input[1, 2], 5)
        self.assertGreater(grad[0].input[0, 0], grad[0].input[0, 2])

    def test_multiple_successive_spikes(self):
        """ """
        # neuron weights, low v_reset only allows one spike per neuron
        params = LIFParameters(v_reset=0.0, v_th=1.0)

        # declare net
        _, apply_fn = serial(
            LIF(
                3,
                n_spikes=5,
                t_max=100e-3,
                params=params,
                mean=0,
                std=0))

        # Same spike time through same input neuron
        weights = np.zeros((2, 3))
        weights = weights.at[0, 0].set(6.5)
        weights = weights.at[0, 2].set(4.5)
        weights = [WeightInput(weights)]

        inputs = EventPropSpike(
            np.array([10e-3]),
            np.array([0]),
            np.array([0.0]))

        res = apply_fn(weights, inputs)[-1]

        # Neuron 0 and 2 should spike at same time
        self.assertLess(res[0].time[1], res[0].time[2])
        self.assertLess(res[0].time[2], res[0].time[3])
        self.assertEqual(res[0].idx[1], res[0].idx[3])
        self.assertNotEqual(res[0].idx[1], res[0].idx[2])

        # now check grads
        def loss_fn(apply_fn, weights, input_spikes):
            ret = apply_fn(weights, input_spikes)
            loss = ret[-1][0].time[2] - 0.5 * ret[-1][0].time[3]
            return loss

        loss_func = partial(loss_fn, apply_fn)

        # check gradients
        _, grad = jax.value_and_grad(loss_func)(weights, inputs)
        self.assertGreater(grad[0].input[0, 0], 0)
        self.assertLess(grad[0].input[0, 2], 0)
        self.assertGreater(grad[0].input[0, 0], grad[0].input[0, 2])


if __name__ == "__main__":
    unittest.main()
