from typing import List

import jax.numpy as np
from jaxsnn.event.dataset.utils import Dataset, add_current
from jaxsnn.event.types import Spike


def constant_dataset(t_max: float, shape: List[int]) -> Dataset:
    inputs = np.array([0.1, 0.2, 1]) * t_max
    target = np.array([0.2, 0.3]) * t_max
    spike_idx = np.array([0, 1, 0])

    input_spikes = Spike(
        np.tile(inputs, (shape + [1])),
        np.tile(spike_idx, (shape + [1])),
    )

    target = np.tile(target, (shape + [1]))
    return (add_current(input_spikes), target, "constant")
