# pylint: disable=invalid-name
import dataclasses
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as np
import tree_math
from jax import random
from jaxsnn.base.params import LIFParameters
from jaxsnn.discrete.threshold import superspike


@dataclasses.dataclass
@tree_math.struct
class LIFInput:
    """Input to a LIF neuron

    Parameters:
        I (jax.Array): membrane input current
        z (jax.Array): input spikes
    """

    I: jax.Array  # pylint: disable=disallowed-name
    z: jax.Array  # pylint: disable=disallowed-name


class LIFState(NamedTuple):
    """State of a LIF neuron

    Parameters:
        z (jax.Array): recurrent spikes
        v (jax.Array): membrane potential
        i (jax.Array): synaptic input current
        input_weights (jax.Array): input weights
        recurrent_weights (jax.Array): recurrentweights
    """

    z: jax.Array
    v: jax.Array
    i: jax.Array


def lif_step(
    init,
    spikes,
    method=superspike,
    params: LIFParameters = LIFParameters(),
    dt=0.001,
):  # pylint: disable=too-many-locals
    state, weights = init
    input_weights, recurrent_weights = weights
    z, v, i = state

    # compute voltage updates
    dv = dt / params.tau_mem * ((params.v_leak - v) + i)
    v_decayed = v + dv

    # compute current updates
    di = -dt / params.tau_syn * i
    i_decayed = i + di

    # compute new spikes
    z_new = method(v_decayed - params.v_th)
    # compute reset
    v_new = (1 - z_new) * v_decayed + z_new * params.v_reset
    # compute current jumps
    i_new = i_decayed + np.matmul(z, recurrent_weights)
    i_new = i_new + np.matmul(spikes, input_weights)

    new_state = LIFState(z_new, v_new, i_new)

    return (new_state, weights), (z_new, new_state)


def LIF(out_dim, method=superspike, scale_in=0.7, scale_rec=0.2):
    """Layer constructor function for a lif (leaky-integrated-fire) layer."""

    def init_fn(rng, input_shape):
        rng, i_key, r_key = random.split(rng, 3)
        input_weights = scale_in * random.normal(i_key, (input_shape, out_dim))
        recurrent_weights = scale_rec * random.normal(
            r_key, (out_dim, out_dim)
        )
        return rng, out_dim, (input_weights, recurrent_weights)

    lif_step_fn = partial(lif_step, method=method)

    def apply_fn(weights, inputs, external, state):  # pylint: disable=unused-argument
        batch = inputs.shape[1]
        shape = (batch, out_dim)
        if state is None:
            layer_index = 0
        else:
            layer_index = state
        this_layer_weights = weights[layer_index]
        s = LIFState(np.zeros(shape), np.zeros(shape), np.zeros(shape))
        _, (output, recording) = jax.lax.scan(
            lif_step_fn, (s, this_layer_weights), inputs
        )
        layer_index += 1
        return layer_index, this_layer_weights, output, recording

    return init_fn, apply_fn
