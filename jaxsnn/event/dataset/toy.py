from typing import List, Tuple

import jax.numpy as np
from jax import random

from jaxsnn.base.types import Array, Spike


def constant_dataset(t_max: float, shape: List[int]) -> Tuple[Spike, Array]:
    input = np.array([0.1, 0.2, 1]) * t_max
    target = np.array([0.2, 0.3]) * t_max  # type: ignore
    spike_idx = np.array([0, 1, 0])

    input_spikes = Spike(
        np.tile(input, (shape + [1])), np.tile(spike_idx, (shape + [1]))
    )

    target = np.tile(target, (shape + [1]))
    return (input_spikes, target)


def linear_dataset(
    rng: random.KeyArray, t_max: float, shape: List[int]
) -> Tuple[Spike, Array]:
    scaling = t_max / 4
    size = np.prod(np.array(shape))
    input = random.uniform(rng, (size, 2))
    encoding = np.array([[0.4, 0.8], [0.8, 0.4]]) * scaling  # type: ignore

    which_class = (input[:, 0] < input[:, 1]).astype(int)
    target = encoding[which_class]

    spike_idx = np.array([0, 1, 3, 4])

    input = np.hstack((input, 1 - input))
    input = input * scaling
    input_spikes = Spike(
        input.reshape(*(shape + [-1])), np.tile(spike_idx, (shape + [1]))
    )

    target = target.reshape(*(shape + [2]))
    return (input_spikes, target)


def circle_dataset(
    rng: random.KeyArray, t_max: float, shape: List[int]
) -> Tuple[Spike, Array]:
    scaling = t_max / 2
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

    spike_idx = np.array([0, 1, 2, 3])

    input = np.hstack((input, 1 - input)) * scaling
    input_spikes = Spike(
        input.reshape(*(shape + [-1])), np.tile(spike_idx, (shape + [1]))
    )

    target = target.reshape(*(shape + [2]))
    return (input_spikes, target)
