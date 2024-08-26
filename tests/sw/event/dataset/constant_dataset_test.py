from jaxsnn.event.dataset import constant_dataset
import unittest


class TestEventDatasetConstantDataset(unittest.TestCase):
    def test_constant_dataset(self):
        dataset = constant_dataset(1e-2, 1000)
        assert dataset[0].shape == (1000, 3)
        assert dataset[1].shape == (1000, 2)


if __name__ == '__main__':
    unittest.main()
