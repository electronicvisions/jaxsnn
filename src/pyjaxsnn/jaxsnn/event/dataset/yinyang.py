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


def good_params_for_hw(params: LIFParameters) -> Dict:
    return {
        "mirror": True,
        "bias_spike": 0.0,
        "correct_target_time": 0.9 * params.tau_syn,
        "wrong_target_time": 1.1 * params.tau_syn,
        "t_late": 2.0 * params.tau_syn,
        "duplication": 5,
        "duplicate_neurons": True,
    }


def yinyang_dataset(  # pylint: disable=too-many-arguments,too-many-locals
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

    inputs = random.uniform(rng, (size * 10 + 100, 2)) * 2.0 * r_big
    which_class = get_class_batched(inputs, r_big, r_small)

    n_per_class = [size // 3, size // 3, size - 2 * (size // 3)]
    idx = np.concatenate(
        [np.where(which_class == i)[0][:n] for i, n in enumerate(n_per_class)]
    )
    idx = random.permutation(subkey, idx, axis=0)
    inputs = inputs[idx]
    which_class = which_class[idx]
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

    spike_idx = np.tile(spike_idx, (np.prod(np.array(shape)), 1))
    assert spike_idx.shape == inputs.shape

    # sort spikes
    sort_idx = np.argsort(inputs, axis=-1)
    inputs = inputs[np.arange(inputs.shape[0])[:, None], sort_idx]
    spike_idx = spike_idx[np.arange(spike_idx.shape[0])[:, None], sort_idx]

    inputs = inputs * t_late
    input_spikes = Spike(
        inputs.reshape(*(shape + [-1])), spike_idx.reshape(*(shape) + [-1])
    )

    target = target.reshape(*(shape + [3]))
    return (add_current(input_spikes), target, "yinyang")
