import jax.numpy as jnp
from jax.nn import log_softmax


def MaxOverTimeDecode():
    """Layer constructor function for a li (leaky-integrated) layer."""

    def init_fun(rng, input_shape):
        return (input_shape, None)

    def apply_fun(params, inputs, **kwargs):
        x = jnp.max(inputs, 0)
        log_p_y = log_softmax(x, axis=1)
        return log_p_y

    return init_fun, apply_fun
