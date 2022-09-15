import jax.numpy as jnp
from jax import random
from jax.lax import scan
from jaxsnn.functional.leaky_integrator import LIState, li_feed_forward_step


def LI(out_dim, scale_in=0.2):
    """Layer constructor function for a li (leaky-integrated) layer."""

    def init_fn(rng, input_shape):
        i_key, r_key = random.split(rng)
        input_weights = scale_in * random.normal(i_key, (input_shape, out_dim))
        return out_dim, input_weights

    def apply_fn(params, inputs, **kwargs):
        batch = inputs.shape[1]
        shape = (batch, out_dim)
        state = LIState(jnp.zeros(shape), jnp.zeros(shape))
        _, voltages = scan(li_feed_forward_step, (state, params), inputs)
        return voltages

    return init_fn, apply_fn
