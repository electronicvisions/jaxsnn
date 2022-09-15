from jax import jit, random
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
        for fun, param in zip(apply_fns, params):
            input = fun(param, input, **kwargs)
            if kwargs.get("recording"):
                recording.append(input)
        return input, recording

    return init_fn, apply_fn
