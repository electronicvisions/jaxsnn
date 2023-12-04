import numpy as np
from jaxsnn.event.dataset import constant_dataset
import unittest


class TestEventDatasetConstantDataset(unittest.TestCase):
    def test_constant_dataset(self):
        dataset = constant_dataset(1e-2, shape=[1000])
        assert dataset[0].idx.shape == (1000, 3)
        assert dataset[0].time.shape == (1000, 3)
        assert dataset[1].shape == (1000, 2)

        dataset = constant_dataset(1e-2, shape=[100, 10])
        assert dataset[0].idx.shape == (100, 10, 3)
        assert dataset[0].time.shape == (100, 10, 3)
        assert dataset[1].shape == (100, 10, 2)


if __name__ == '__main__':
    unittest.main()
