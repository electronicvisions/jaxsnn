from typing import List, Optional

import jax.numpy as np
from jax import random
from jaxsnn.event.dataset.utils import Dataset, add_current
from jaxsnn.event.types import Spike


def circle_dataset(  # pylint: disable=too-many-arguments,too-many-locals
    rng: random.KeyArray,
    tau_syn: float,
    shape: List[int],
    mirror: bool = True,
    bias_spike: Optional[float] = 0.0,
    duplication: Optional[int] = None,
) -> Dataset:
    scaling = 1.5 * tau_syn
    size = np.prod(np.array(shape))
    inputs = random.uniform(rng, (size, 2))
    encoding = np.array([[0, 1], [1, 0]]) * scaling

    # determine class
    center = (0.5, 0.5)
    radius = np.sqrt(0.5 / np.pi)  # spread classes equal
    which_class = (
        ((inputs[:, 0] - center[0]) ** 2 + (inputs[:, 1] - center[1]) ** 2)
        <= radius**2
    ).astype(int)
    target = encoding[which_class]

    spike_idx = np.array([0, 1])
    if mirror:
        spike_idx = np.concatenate((spike_idx, np.array([2, 3])))
        inputs = np.hstack((inputs, 1 - inputs))

    if bias_spike is not None:
        spike_idx = np.concatenate((spike_idx, np.array([spike_idx[-1] + 1])))
        column = np.full(size, bias_spike)[:, None]
        inputs = np.hstack((inputs, column))

    if duplication is not None:
        spike_idx = np.repeat(spike_idx, duplication, axis=-1)
        inputs = np.repeat(inputs, duplication, axis=-1)

    inputs = inputs * scaling

    spike_idx = np.tile(spike_idx, (np.prod(np.array(shape)), 1))
    assert spike_idx.shape == inputs.shape

    # sort spikes
    sort_idx = np.argsort(inputs, axis=-1)
    inputs = inputs[np.arange(inputs.shape[0])[:, None], sort_idx]
    spike_idx = spike_idx[np.arange(spike_idx.shape[0])[:, None], sort_idx]

    input_spikes = Spike(
        inputs.reshape(*(shape + [-1])),
        spike_idx.reshape(*(shape) + [-1]),
    )

    target = target.reshape(*(shape + [2]))
    return (add_current(input_spikes), target, "circle")
