import numpy as np
from jax import random
from jaxsnn.event.dataset import linear_dataset
import unittest


class TestEventDatasetLinearDataset(unittest.TestCase):
    def test_linear_dataset(self):
        rng = random.PRNGKey(42)
        dataset = linear_dataset(
            rng, 1e-2, shape=[1000], mirror=False, bias_spike=None
        )
        assert dataset[0].idx.shape == (1000, 2)
        assert dataset[0].time.shape == (1000, 2)
        assert dataset[1].shape == (1000, 2)

        dataset = linear_dataset(
            rng, 1e-2, shape=[1000], mirror=True, bias_spike=None
        )
        assert dataset[0].idx.shape == (1000, 4)
        assert dataset[0].time.shape == (1000, 4)
        assert dataset[1].shape == (1000, 2)

        dataset = linear_dataset(
            rng, 1e-2, shape=[1000], mirror=True, bias_spike=0.0
        )
        assert dataset[0].idx.shape == (1000, 5)
        assert dataset[0].time.shape == (1000, 5)
        assert dataset[1].shape == (1000, 2)

        dataset = linear_dataset(
            rng, 1e-2, shape=[100, 10], mirror=True, bias_spike=0.0
        )
        assert dataset[0].idx.shape == (100, 10, 5)
        assert dataset[0].time.shape == (100, 10, 5)
        assert dataset[1].shape == (100, 10, 2)

        # test sorted
        assert np.all(dataset[0].time == np.sort(dataset[0].time, axis=-1))


if __name__ == '__main__':
    unittest.main()
