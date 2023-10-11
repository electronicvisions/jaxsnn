from typing import List, Optional

import jax.numpy as np
from jax import random
from jaxsnn.event.types import Spike
from jaxsnn.event.dataset.utils import Dataset, add_current


def circle_dataset(
    rng: random.KeyArray,
    tau_syn: float,
    shape: List[int],
    mirror: bool = True,
    bias_spike: Optional[float] = 0.0,
    duplication: Optional[int] = None,
) -> Dataset:
    scaling = 1.5 * tau_syn
    size = np.prod(np.array(shape))
    input = random.uniform(rng, (size, 2))
    encoding = np.array([[0, 1], [1, 0]]) * scaling  # type: ignore

    # determine class
    center = (0.5, 0.5)
    radius = np.sqrt(0.5 / np.pi)  # spread classes equal
    which_class = (
        (input[:, 0] - center[0]) ** 2 + (input[:, 1] - center[1]) ** 2
        <= radius**2
    ).astype(int)
    target = encoding[which_class]

    spike_idx = np.array([0, 1])
    if mirror:
        spike_idx = np.concatenate((spike_idx, np.array([2, 3])))
        input = np.hstack((input, 1 - input))

    if bias_spike is not None:
        spike_idx = np.concatenate((spike_idx, np.array([spike_idx[-1] + 1])))
        column = np.full(size, bias_spike)[:, None]
        input = np.hstack((input, column))

    if duplication is not None:
        spike_idx = np.repeat(spike_idx, duplication, axis=-1)
        input = np.repeat(input, duplication, axis=-1)

    input = input * scaling

    spike_idx = np.tile(spike_idx, (np.prod(np.array(shape)), 1))
    assert spike_idx.shape == input.shape

    # sort spikes
    sort_idx = np.argsort(input, axis=-1)
    input = input[np.arange(input.shape[0])[:, None], sort_idx]
    spike_idx = spike_idx[np.arange(spike_idx.shape[0])[:, None], sort_idx]

    input_spikes = Spike(
        input.reshape(*(shape + [-1])),
        spike_idx.reshape(*(shape) + [-1]),
    )

    target = target.reshape(*(shape + [2]))
    return (add_current(input_spikes), target, "circle")
