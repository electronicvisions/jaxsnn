from jaxsnn.functional.encode import spatio_temporal_encode


def SpatioTemporalEncode(T, t_late, DT):
    """Layer constructor function for a li (leaky-integrated) layer."""

    def init_fun(rng, input_shape):
        return (input_shape, None)

    def apply_fun(params, inputs, **kwargs):
        return spatio_temporal_encode(inputs, T, t_late, DT)

    return init_fun, apply_fun
