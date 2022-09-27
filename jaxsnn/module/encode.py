from jaxsnn.functional.encode import spatio_temporal_encode


def SpatioTemporalEncode(T, t_late, DT):
    def init_fn(rng, input_shape):
        return (input_shape, None, rng)

    def apply_fn(params, inputs, **kwargs):
        return spatio_temporal_encode(inputs, T, t_late, DT), None

    return init_fn, apply_fn
