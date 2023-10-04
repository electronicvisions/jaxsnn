from typing import List, Tuple, Optional

import jax.numpy as np
from jax import random

from jaxsnn.base.types import Array, EventPropSpike, Spike
from jaxsnn.dataset.yinyang import get_class_batched

Dataset = Tuple[EventPropSpike, Array, str]


def add_current(spike: Spike) -> EventPropSpike:
    return EventPropSpike(
        spike.time, spike.idx, np.zeros_like(spike.idx, dtype=spike.time.dtype)
    )


def constant_dataset(t_max: float, shape: List[int]) -> Dataset:
    input = np.array([0.1, 0.2, 1]) * t_max
    target = np.array([0.2, 0.3]) * t_max  # type: ignore
    spike_idx = np.array([0, 1, 0])

    input_spikes = Spike(
        np.tile(input, (shape + [1])),
        np.tile(spike_idx, (shape + [1])),
    )

    target = np.tile(target, (shape + [1]))
    return (add_current(input_spikes), target, "constant")


def linear_dataset(
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
    input = random.uniform(rng, (size, 2))
    encoding = np.array(
        [
            [correct_target_time, wrong_target_time],
            [wrong_target_time, correct_target_time],
        ]
    )

    which_class = (input[:, 0] < input[:, 1]).astype(int)
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

    input = input * t_late

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
    return (add_current(input_spikes), target, "linear")


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
        (input[:, 0] - center[0]) ** 2 + (input[:, 1] - center[1]) ** 2 <= radius**2
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


def yinyang_dataset(
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
    rng, subkey = random.split(rng)
    r_big = 0.5
    r_small = 0.1
    size = np.prod(np.array(shape))

    if correct_target_time is None:
        correct_target_time = 0.9 * t_late
    if wrong_target_time is None:
        wrong_target_time = 1.5 * t_late

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
