from jaxsnn.functional.decode import decode


def MaxOverTimeDecode():
    def init_fn(rng, input_shape):
        return (input_shape, None, rng)

    def apply_fn(params, inputs, **kwargs):
        return decode(inputs), None

    return init_fn, apply_fn
