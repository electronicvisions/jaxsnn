# pylint: disable=invalid-name
import dataclasses

import jax
import jax.numpy as np
import tree_math
from jax import random
from jaxsnn.base.params import LIParameters


@dataclasses.dataclass
@tree_math.struct
class LIState:
    """State of a leaky-integrator

    Parameters:
        v (jax.Array): membrane voltage
        i (jax.Array): input current
    """

    v: jax.Array
    i: jax.Array


@jax.jit
def li_feed_forward_step(
    init,
    spikes: jax.Array,
    params: LIParameters = LIParameters(),
    dt: float = 0.001,
):
    state, input_weights = init
    # compute current jumps
    i_jump = state.i + np.matmul(spikes, input_weights)
    # compute voltage updates
    dv = dt * params.tau_mem_inv * ((params.v_leak - state.v) + i_jump)
    v_new = state.v + dv

    # compute current updates
    di = -dt * params.tau_syn_inv * i_jump
    i_decayed = i_jump + di

    return (LIState(v_new, i_decayed), input_weights), v_new


def li_integrate(init, spikes):
    return jax.lax.scan(li_feed_forward_step, init, spikes)


def LI(out_dim, scale_in=0.2):
    """Layer constructor function for a li (leaky-integrated) layer."""

    def init_fn(rng, input_shape):
        rng, i_key = random.split(rng)
        input_weights = scale_in * random.normal(i_key, (input_shape, out_dim))
        return out_dim, input_weights, rng

    def apply_fn(weights, inputs, **kwargs):  # pylint: disable=unused-argument
        batch = inputs.shape[1]
        shape = (batch, out_dim)
        state = LIState(np.zeros(shape), np.zeros(shape))
        _, voltages = jax.lax.scan(
            li_feed_forward_step, (state, weights), inputs
        )
        return voltages

    return init_fn, apply_fn


def LIStep(out_dim, scale_in=0.2):
    """Layer constructor function for a li (leaky-integrated) layer."""

    def init_fn(rng, input_shape):
        rng, i_key = random.split(rng)
        input_weights = scale_in * random.normal(i_key, (input_shape, out_dim))
        return out_dim, input_weights, rng

    def apply_fn(
        state, weights, inputs, **kwargs
    ):  # pylint: disable=unused-argument
        return li_feed_forward_step((state, weights), inputs)

    def state_fn(batch_size):
        shape = (batch_size, out_dim)
        state = LIState(np.zeros(shape), np.zeros(shape))
        return state

    return init_fn, apply_fn, state_fn
