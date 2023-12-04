from typing import List, Optional

import jax.numpy as np
from jax import random
from jaxsnn.event.dataset.utils import Dataset, add_current
from jaxsnn.event.types import Spike


def linear_dataset(  # pylint: disable=too-many-arguments,too-many-locals
    rng: random.KeyArray,
    t_late: float,
    shape: List[int],
    mirror: bool = True,
    bias_spike: Optional[float] = 0.0,
    correct_target_time: Optional[float] = None,
    wrong_target_time: Optional[float] = None,
    duplication: Optional[int] = None,
    duplicate_neurons: bool = False,
) -> Dataset:
    if correct_target_time is None:
        correct_target_time = 0.5 * t_late
    if wrong_target_time is None:
        wrong_target_time = 1.0 * t_late

    size = np.prod(np.array(shape))
    inputs = random.uniform(rng, (size, 2))
    encoding = np.array(
        [
            [correct_target_time, wrong_target_time],
            [wrong_target_time, correct_target_time],
        ]
    )

    which_class = (inputs[:, 0] < inputs[:, 1]).astype(int)
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
        inputs = np.repeat(inputs, duplication, axis=-1)

        if duplicate_neurons:
            # duplicate over multiple neurons
            spike_idx = np.arange(spike_idx.shape[0] * duplication)
        else:
            spike_idx = np.repeat(spike_idx, duplication, axis=-1)

    inputs = inputs * t_late

    spike_idx = np.tile(spike_idx, (np.prod(np.array(shape)), 1))
    assert spike_idx.shape == inputs.shape

    # sort spikes
    sort_idx = np.argsort(inputs, axis=-1)
    inputs = inputs[np.arange(inputs.shape[0])[:, None], sort_idx]
    spike_idx = spike_idx[np.arange(spike_idx.shape[0])[:, None], sort_idx]

    inputs_spikes = Spike(
        inputs.reshape(*(shape + [-1])),
        spike_idx.reshape(*(shape) + [-1]),
    )

    target = target.reshape(*(shape + [2]))
    return (add_current(inputs_spikes), target, "linear")
