from jax import random
from jaxsnn.base.dataset import yinyang_dataset
import unittest


class TestEventDatasetYinYangDataset(unittest.TestCase):
    def test_yinyang_dataset(self):
        rng = random.PRNGKey(42)
        dataset = yinyang_dataset(
            rng,
            size=1000,
            mirror=False,
            bias_spike=None,
        )
        assert dataset[0].shape == (1000, 2)
        assert dataset[1].shape == (1000,)

        dataset = yinyang_dataset(
            rng,
            size=1000,
            mirror=True,
            bias_spike=None,
        )
        assert dataset[0].shape == (1000, 4)
        assert dataset[1].shape == (1000,)

        dataset = yinyang_dataset(
            rng,
            size=1000,
            mirror=True,
            bias_spike=0.0,
        )
        assert dataset[0].shape == (1000, 5)
        assert dataset[1].shape == (1000,)


if __name__ == '__main__':
    unittest.main()
