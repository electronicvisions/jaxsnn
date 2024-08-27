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

    new_state = LIState(v_new, i_decayed)

    return (new_state, input_weights), (v_new, new_state)


def LI(out_dim, scale_in=0.2):
    """Layer constructor function for a li (leaky-integrated) layer."""

    def init_fn(rng, input_shape):
        rng, layer_rng = jax.random.split(rng)
        input_weights = scale_in * random.normal(
            layer_rng, (input_shape, out_dim))
        return rng, out_dim, input_weights

    def apply_fn(weights, inputs, external, state):  # pylint: disable=unused-argument
        if state is None:
            layer_index = 0
        else:
            layer_index = state
        this_layer_weights = weights[layer_index]
        batch = inputs.shape[1]
        shape = (batch, out_dim)
        s = LIState(np.zeros(shape), np.zeros(shape))
        _, (voltages, recording) = jax.lax.scan(
            li_feed_forward_step, (s, this_layer_weights), inputs
        )
        layer_index += 1
        return layer_index, this_layer_weights, voltages, recording

    return init_fn, apply_fn
