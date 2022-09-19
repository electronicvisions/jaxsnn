from jax import random
from jax.lax import scan


def serial(*layers):
    """Combinator for composing layers in serial.
    Args:
      *layers: a sequence of layers, each an (init_fn, apply_fn) double.
    Returns:
      A new layer, meaning an (init_fn, apply_fn) double, representing the serial
      composition of the given sequence of layers.
    """
    init_fns, apply_fns = zip(*layers)

    def init_fn(rng, input_shape):
        params = []
        for init_fn in init_fns:
            rng, layer_rng = random.split(rng)
            input_shape, param = init_fn(layer_rng, input_shape)
            params.append(param)
        return input_shape, params

    # @jit
    def apply_fn(params, input, **kwargs):
        recording = []
        for fn, param in zip(apply_fns, params):
            input = fn(param, input, **kwargs)
            if kwargs.get("recording"):
                recording.append(input)
        return input, recording

    return init_fn, apply_fn


def euler_integrate(*layers):
    init_fns, apply_fns, state_fns = zip(*layers)

    def init_fn(rng, input_shape):
        params = []
        for init_fn in init_fns:
            rng, layer_rng = random.split(rng)
            input_shape, param = init_fn(layer_rng, input_shape)
            params.append(param)
        return input_shape, params

    def apply_fn(params, input, **kwargs):
        batch_size = input.shape[1]
        states = [state_fn(batch_size) for state_fn in state_fns]

        def inner(states, input_ts):
            new_states = []
            for fn, param, state in zip(apply_fns, params, states):
                (new_state, weights), input_ts = fn(state, param, input_ts, **kwargs)
                new_states.append(new_state)
            return new_states, input_ts

        _, output = scan(inner, states, input)
        return output

    return init_fn, apply_fn


# where is batch size?
# aktuell (70, 3)
