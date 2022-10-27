import jax.numpy as np
from jax import random
from jax.lax import scan
from jaxsnn.functional.leaky_integrator import LIState, li_feed_forward_step


def LI(out_dim, scale_in=0.2):
    """Layer constructor function for a li (leaky-integrated) layer."""

    def init_fn(rng, input_shape):
        rng, i_key = random.split(rng)
        input_weights = scale_in * random.normal(i_key, (input_shape, out_dim))
        return out_dim, input_weights, rng

    def apply_fn(params, inputs, **kwargs):
        batch = inputs.shape[1]
        shape = (batch, out_dim)
        state = LIState(np.zeros(shape), np.zeros(shape))
        _, voltages = scan(li_feed_forward_step, (state, params), inputs)
        return voltages

    return init_fn, apply_fn


def LIStep(out_dim, scale_in=0.2):
    """Layer constructor function for a li (leaky-integrated) layer."""

    def init_fn(rng, input_shape):
        rng, i_key = random.split(rng)
        input_weights = scale_in * random.normal(i_key, (input_shape, out_dim))
        return out_dim, input_weights, rng

    def apply_fn(state, params, inputs, **kwargs):
        return li_feed_forward_step((state, params), inputs)

    def state_fn(batch_size):
        shape = (batch_size, out_dim)
        state = LIState(np.zeros(shape), np.zeros(shape))
        return state

    return init_fn, apply_fn, state_fn
