from jaxsnn.base.types import Weight, Spike

import jax

from typing import List


def serial(*layers):
    init_fns, apply_fns = zip(*layers)

    def init_fn(rng: jax.random.KeyArray, input_shape: int) -> List[Weight]:
        params = []
        for init_fn in init_fns:
            if len(init_fns) > 1:
                rng, layer_rng = jax.random.split(rng)
            else:
                layer_rng = rng
            input_shape, param = init_fn(layer_rng, input_shape)
            params.append(param)
        return params

    def apply_fn(params: List[Weight], spikes: Spike):
        recording = []
        layer_start = 0
        for fn, param in zip(apply_fns, params):
            layer_start += param.input.shape[0]
            spikes = fn(layer_start, param, spikes)
            recording.append(spikes)
        return recording

    return init_fn, apply_fn


def serial_spikes_known(*layers):
    init_fns, apply_fns = zip(*layers)

    def init_fn(rng: jax.random.KeyArray, input_shape: int) -> List[Weight]:
        params = []
        for init_fn in init_fns:
            if len(init_fns) > 1:
                rng, layer_rng = jax.random.split(rng)
            else:
                layer_rng = rng
            input_shape, param = init_fn(layer_rng, input_shape)
            params.append(param)
        return params

    def apply_fn(known_spikes: List[Spike], params: List[Weight], spikes: Spike):
        recording = []
        layer_start = 0
        for fn, param, known in zip(apply_fns, params, known_spikes):
            layer_start += param.input.shape[0]
            spikes = fn(layer_start, param, spikes, known)
            recording.append(spikes)
        return recording

    return init_fn, apply_fn
