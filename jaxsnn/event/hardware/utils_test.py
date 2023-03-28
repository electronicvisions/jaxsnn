from jaxsnn.event.hardware.utils import add_noise_batch, sort_batch
from jaxsnn.base.types import Spike
import jax.numpy as np
from jax import random
from numpy.testing import assert_array_equal


def test_add_noise_batch():
    spikes = Spike(
        idx=np.expand_dims(np.arange(10), axis=0),
        time=np.expand_dims(np.arange(10), axis=0),
    )
    rng = random.PRNGKey(42)
    with_noise = add_noise_batch(spikes, rng, std=1)
    assert_array_equal(with_noise.idx, np.array([[0, 1, 2, 5, 3, 4, 6, 7, 8, 9]]))

    with_noise = add_noise_batch(spikes, rng, std=3)
    assert_array_equal(with_noise.idx, np.array([[2, 1, 0, 5, 6, 7, 3, 4, 8, 9]]))


def test_sort_batch():
    spikes = Spike(
        idx=np.expand_dims(np.arange(10), axis=0),
        time=np.expand_dims(np.arange(9, -1, -1), axis=0),
    )
    print(spikes.idx.shape)
    print(spikes.time.shape)
    sorted = sort_batch(spikes)
    print(sorted.idx.shape)
    print(sorted.time.shape)
    assert_array_equal(sorted.idx, np.expand_dims(np.arange(9, -1, -1), axis=0))
