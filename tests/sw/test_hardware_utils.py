import jax.numpy as jnp
from jax import random
from jaxsnn.event.hardware.utils import add_noise_batch, sort_batch
from jaxsnn.event.types import Spike
from numpy.testing import assert_array_equal
import unittest


class TestEventHwUtils(unittest.TestCase):
    def test_add_noise_batch(self):
        spikes = Spike(
            idx=jnp.expand_dims(jnp.arange(10), axis=0),
            time=jnp.expand_dims(jnp.arange(10), axis=0))

        rng = random.PRNGKey(42)
        with_noise = add_noise_batch(spikes, rng, std=1)
        self.assertIsNone(
            assert_array_equal(
                with_noise.idx, jnp.array([[0, 1, 2, 5, 3, 4, 6, 7, 8, 9]])))

        with_noise = add_noise_batch(spikes, rng, std=3)
        self.assertIsNone(
            assert_array_equal(
                with_noise.idx, jnp.array([[2, 1, 0, 5, 6, 7, 3, 4, 8, 9]])))

    def test_sort_batch(self):
        spikes = Spike(
            idx=jnp.expand_dims(jnp.arange(10), axis=0),
            time=jnp.expand_dims(jnp.arange(9, -1, -1), axis=0))
        sorted = sort_batch(spikes)
        self.assertIsNone(
            assert_array_equal(
                sorted.idx, jnp.expand_dims(jnp.arange(9, -1, -1), axis=0)))


if __name__ == '__main__':
    unittest.main()
