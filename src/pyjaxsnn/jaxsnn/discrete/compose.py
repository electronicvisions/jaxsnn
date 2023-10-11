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


def euler_integrate(*layers):
    init_fns, apply_fns, state_fns = zip(*layers)

    def init_fn(rng, input_shape):
        weights = []
        for init_fn in init_fns:
            input_shape, param, rng = init_fn(rng, input_shape)
            weights.append(param)
        return input_shape, weights, rng

    def apply_fn(weights, inputs, **kwargs):  # pylint: disable=unused-argument
        batch_size = inputs.shape[1]
        states = [state_fn(batch_size) for state_fn in state_fns]

        def inner(states, input_ts, **kwargs):
            new_states = []
            for layer_apply_fn, param, state in zip(
                apply_fns, weights, states
            ):
                (new_state, _), input_ts = layer_apply_fn(
                    state, param, input_ts, **kwargs
                )
                new_states.append(new_state)

            return new_states, (input_ts, new_states)

        _, (output, recording) = jax.lax.scan(inner, states, inputs)
        return output, recording

    return init_fn, apply_fn
