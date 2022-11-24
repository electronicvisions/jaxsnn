from jax import random

from jaxsnn.event.dataset import linear_dataset, constant_dataset, circle_dataset


def test_linear_dataset():
    rng = random.PRNGKey(42)
    dataset = linear_dataset(rng, 1e-2, shape=[1000])
    assert dataset[0].idx.shape == (1000, 4)
    assert dataset[0].time.shape == (1000, 4)
    assert dataset[1].shape == (1000, 2)

    dataset = linear_dataset(rng, 1e-2, shape=[100, 10])
    assert dataset[0].idx.shape == (100, 10, 4)
    assert dataset[0].time.shape == (100, 10, 4)
    assert dataset[1].shape == (100, 10, 2)


def test_circle_dataset():
    rng = random.PRNGKey(42)
    dataset = circle_dataset(rng, 1e-2, shape=[1000])
    assert dataset[0].idx.shape == (1000, 4)
    assert dataset[0].time.shape == (1000, 4)
    assert dataset[1].shape == (1000, 2)

    dataset = circle_dataset(rng, 1e-2, shape=[100, 10])
    assert dataset[0].idx.shape == (100, 10, 4)
    assert dataset[0].time.shape == (100, 10, 4)
    assert dataset[1].shape == (100, 10, 2)


def test_constant_dataset():
    dataset = constant_dataset(1e-2, shape=[1000])
    assert dataset[0].idx.shape == (1000, 3)
    assert dataset[0].time.shape == (1000, 3)
    assert dataset[1].shape == (1000, 2)

    dataset = constant_dataset(1e-2, shape=[100, 10])
    assert dataset[0].idx.shape == (100, 10, 3)
    assert dataset[0].time.shape == (100, 10, 3)
    assert dataset[1].shape == (100, 10, 2)
