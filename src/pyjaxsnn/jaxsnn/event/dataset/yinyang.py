from typing import Dict, List, Optional

import jax.numpy as np
from jax import random
from jaxsnn.base.params import LIFParameters
from jaxsnn.discrete.dataset.yinyang import get_class_batched
from jaxsnn.event.dataset.utils import Dataset, add_current
from jaxsnn.event.types import Spike


def good_params(params: LIFParameters) -> Dict:
    return {
        "mirror": True,
        "bias_spike": 0.0,
        "correct_target_time": 0.9 * params.tau_syn,
        "wrong_target_time": 1.1 * params.tau_syn,
        "t_late": 2.0 * params.tau_syn,
    }


def yinyang_dataset(
    rng: random.KeyArray,
    shape: List[int],
    t_late: float,
    correct_target_time: float,
    wrong_target_time: float,
    mirror: bool = True,
    bias_spike: Optional[float] = 0.0,
    duplication: Optional[int] = None,
    duplicate_neurons: bool = False,
) -> Dataset:
    rng, subkey = random.split(rng)
    r_big = 0.5
    r_small = 0.1
    size = np.prod(np.array(shape))

    encoding = np.array(
        [
            [correct_target_time, wrong_target_time, wrong_target_time],
            [wrong_target_time, correct_target_time, wrong_target_time],
            [wrong_target_time, wrong_target_time, correct_target_time],
        ]
    )

    input = random.uniform(rng, (size * 10 + 100, 2)) * 2.0 * r_big
    which_class = get_class_batched(input, r_big, r_small)

    n_per_class = [size // 3, size // 3, size - 2 * (size // 3)]
    idx = np.concatenate(
        [np.where(which_class == i)[0][:n] for i, n in enumerate(n_per_class)]
    )
    idx = random.permutation(subkey, idx, axis=0)
    input = input[idx]
    which_class = which_class[idx]
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
        input = np.repeat(input, duplication, axis=-1)
        if duplicate_neurons:
            # duplicate over multiple neurons
            spike_idx = np.arange(spike_idx.shape[0] * duplication)
        else:
            spike_idx = np.repeat(spike_idx, duplication, axis=-1)

    spike_idx = np.tile(spike_idx, (np.prod(np.array(shape)), 1))
    assert spike_idx.shape == input.shape

    # sort spikes
    sort_idx = np.argsort(input, axis=-1)
    input = input[np.arange(input.shape[0])[:, None], sort_idx]
    spike_idx = spike_idx[np.arange(spike_idx.shape[0])[:, None], sort_idx]

    input = input * t_late
    input_spikes = Spike(
        input.reshape(*(shape + [-1])), spike_idx.reshape(*(shape) + [-1])
    )

    target = target.reshape(*(shape + [3]))
    return (add_current(input_spikes), target, "yinyang")
