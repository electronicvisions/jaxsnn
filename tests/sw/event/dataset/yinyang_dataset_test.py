import numpy as np
from jax import random
from jaxsnn.event.dataset import yinyang_dataset
import unittest


class TestEventDatasetYinYangDataset(unittest.TestCase):
    def test_yinyang_dataset(self):
        rng = random.PRNGKey(42)
        dataset = yinyang_dataset(
            rng,
            shape=[1000],
            t_late=1e-2,
            t_correct_target=1e-3,
            t_wrong_target=5e-3,
            mirror=False,
            t_bias=None,
        )
        assert dataset[0].idx.shape == (1000, 2)
        assert dataset[0].time.shape == (1000, 2)
        assert dataset[1].shape == (1000, 3)

        dataset = yinyang_dataset(
            rng,
            shape=[100, 10],
            t_late=1e-2,
            t_correct_target=1e-3,
            t_wrong_target=5e-3,
            mirror=True,
            t_bias=None,
        )
        assert dataset[0].idx.shape == (100, 10, 4)
        assert dataset[0].time.shape == (100, 10, 4)
        assert dataset[1].shape == (100, 10, 3)

        dataset = yinyang_dataset(
            rng,
            shape=[100, 10],
            t_late=1e-2,
            t_correct_target=1e-3,
            t_wrong_target=5e-3,
            mirror=True,
            t_bias=0.0,
        )
        assert dataset[0].idx.shape == (100, 10, 5)
        assert dataset[0].time.shape == (100, 10, 5)
        assert dataset[1].shape == (100, 10, 3)

        # test sorted
        assert np.all(dataset[0].time == np.sort(dataset[0].time, axis=-1))


if __name__ == '__main__':
    unittest.main()
