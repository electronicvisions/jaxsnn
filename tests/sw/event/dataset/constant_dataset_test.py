import numpy as np
from jaxsnn.event.dataset import constant_dataset


def test_constant_dataset():
    dataset = constant_dataset(1e-2, shape=[1000])
    assert dataset[0].idx.shape == (1000, 3)
    assert dataset[0].time.shape == (1000, 3)
    assert dataset[1].shape == (1000, 2)

    dataset = constant_dataset(1e-2, shape=[100, 10])
    assert dataset[0].idx.shape == (100, 10, 3)
    assert dataset[0].time.shape == (100, 10, 3)
    assert dataset[1].shape == (100, 10, 2)
