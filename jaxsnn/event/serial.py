from jaxsnn.base.types import (
    Spike,
    Weight,  # TODO: This is not sufficiently generic to be there
)

import jax
import jax.numpy as np

from typing import List


def reorder_spikes(spikes: Spike, t_max: float) -> Spike:
    # move input spikes (-1) of previous layer to the end and replace times by t_max
    times = np.where(spikes.idx == -1, t_max, spikes.time)
    ordering = np.argsort(times)
    spikes = Spike(times[ordering], spikes.idx[ordering])
    return spikes


def serial(*layers):
    init_fns, apply_fns = zip(*layers)

    def init_fn(rng: jax.random.KeyArray, input_shape: int) -> List[Weight]:
        params = []
        for init_fn in init_fns:
            rng, layer_rng = jax.random.split(rng)
            input_shape, param = init_fn(layer_rng, input_shape)
            params.append(param)
        return params

    def apply_fn(params, spikes):
        recording = []
        for fn, param in zip(apply_fns, params):
            spikes = fn(param, reorder_spikes(spikes))
            recording.append(spikes)
        return recording

    return init_fn, apply_fn
