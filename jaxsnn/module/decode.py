import jax.numpy as jnp
from jax import jit
from jax.nn import log_softmax
from jaxsnn.functional.decode import decode


def MaxOverTimeDecode():
    """Layer constructor function for a li (leaky-integrated) layer."""

    def init_fn(rng, input_shape):
        return (input_shape, None)

    def apply_fn(params, inputs, **kwargs):
        return decode(inputs)

    return init_fn, apply_fn
