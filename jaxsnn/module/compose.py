from jax import random


def serial(*layers):
    """Combinator for composing layers in serial.
    Args:
      *layers: a sequence of layers, each an (init_fun, apply_fun) double.
    Returns:
      A new layer, meaning an (init_fun, apply_fun) double, representing the serial
      composition of the given sequence of layers.
    """
    init_funs, apply_funs = zip(*layers)

    def init_fun(rng, input_shape):
        params = []
        for init_fun in init_funs:
            rng, layer_rng = random.split(rng)
            input_shape, param = init_fun(layer_rng, input_shape)
            params.append(param)
        return input_shape, params

    def apply_fun(params, inputs, **kwargs):
        for fun, param in zip(apply_funs, params):
            inputs = fun(param, inputs, **kwargs)
        return inputs

    return init_fun, apply_fun
