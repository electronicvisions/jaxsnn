import jax


def serial(*layers):
    """Combinator for composing layers in serial.
    Args:
      *layers: a sequence of layers, each an (init_fn, apply_fn) double.
    Returns:
      A new layer, meaning an (init_fn, apply_fn) double, representing the
      serial composition of the given sequence of layers.
    """
    init_fns, apply_fns = zip(*layers)

    def init_fn(rng, input_shape):
        weights = []
        for init_fn in init_fns:
            input_shape, param, rng = init_fn(rng, input_shape)
            weights.append(param)
        return input_shape, weights

    @jax.jit
    def apply_fn(weights, inputs, **kwargs):
        recording = []
        for layer_apply_fn, param in zip(apply_fns, weights):
            inputs, layer_records = layer_apply_fn(param, inputs, **kwargs)
            if isinstance(layer_records, list):
                recording.extend(layer_records)
            else:
                recording.append(layer_records)
        return inputs, recording

    return init_fn, apply_fn
