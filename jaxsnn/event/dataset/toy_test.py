from jax import random
import numpy as np
from jaxsnn.event.dataset import (
    linear_dataset,
    constant_dataset,
    circle_dataset,
    yinyang_dataset,
)


def test_linear_dataset():
    rng = random.PRNGKey(42)
    dataset = linear_dataset(rng, 1e-2, shape=[1000], mirror=False, bias_spike=None)
    assert dataset[0].idx.shape == (1000, 2)
    assert dataset[0].time.shape == (1000, 2)
    assert dataset[1].shape == (1000, 2)

    dataset = linear_dataset(rng, 1e-2, shape=[1000], mirror=True, bias_spike=None)
    assert dataset[0].idx.shape == (1000, 4)
    assert dataset[0].time.shape == (1000, 4)
    assert dataset[1].shape == (1000, 2)

    dataset = linear_dataset(rng, 1e-2, shape=[1000], mirror=True, bias_spike=0.0)
    assert dataset[0].idx.shape == (1000, 5)
    assert dataset[0].time.shape == (1000, 5)
    assert dataset[1].shape == (1000, 2)

    dataset = linear_dataset(rng, 1e-2, shape=[100, 10], mirror=True, bias_spike=0.0)
    assert dataset[0].idx.shape == (100, 10, 5)
    assert dataset[0].time.shape == (100, 10, 5)
    assert dataset[1].shape == (100, 10, 2)

    # test sorted
    assert np.all(dataset[0].time == np.sort(dataset[0].time, axis=-1))


def test_circle_dataset():
    rng = random.PRNGKey(42)
    dataset = circle_dataset(rng, 1e-2, shape=[1000], mirror=False, bias_spike=None)
    assert dataset[0].idx.shape == (1000, 2)
    assert dataset[0].time.shape == (1000, 2)
    assert dataset[1].shape == (1000, 2)

    dataset = circle_dataset(rng, 1e-2, shape=[100, 10], mirror=True, bias_spike=None)
    assert dataset[0].idx.shape == (100, 10, 4)
    assert dataset[0].time.shape == (100, 10, 4)
    assert dataset[1].shape == (100, 10, 2)

    dataset = circle_dataset(rng, 1e-2, shape=[100, 10], mirror=True, bias_spike=0.0)
    assert dataset[0].idx.shape == (100, 10, 5)
    assert dataset[0].time.shape == (100, 10, 5)
    assert dataset[1].shape == (100, 10, 2)

    # test sorted
    assert np.all(dataset[0].time == np.sort(dataset[0].time, axis=-1))


def test_yinyang_dataset():
    rng = random.PRNGKey(42)
    dataset = yinyang_dataset(rng, 1e-2, shape=[1000], mirror=False, bias_spike=None)
    assert dataset[0].idx.shape == (1000, 2)
    assert dataset[0].time.shape == (1000, 2)
    assert dataset[1].shape == (1000, 3)

    dataset = yinyang_dataset(rng, 1e-2, shape=[100, 10], mirror=True, bias_spike=None)
    assert dataset[0].idx.shape == (100, 10, 4)
    assert dataset[0].time.shape == (100, 10, 4)
    assert dataset[1].shape == (100, 10, 3)

    dataset = yinyang_dataset(rng, 1e-2, shape=[100, 10], mirror=True, bias_spike=0.0)
    assert dataset[0].idx.shape == (100, 10, 5)
    assert dataset[0].time.shape == (100, 10, 5)
    assert dataset[1].shape == (100, 10, 3)

    # test sorted
    assert np.all(dataset[0].time == np.sort(dataset[0].time, axis=-1))


def test_constant_dataset():
    dataset = constant_dataset(1e-2, shape=[1000])
    assert dataset[0].idx.shape == (1000, 3)
    assert dataset[0].time.shape == (1000, 3)
    assert dataset[1].shape == (1000, 2)

    dataset = constant_dataset(1e-2, shape=[100, 10])
    assert dataset[0].idx.shape == (100, 10, 3)
    assert dataset[0].time.shape == (100, 10, 3)
    assert dataset[1].shape == (100, 10, 2)
