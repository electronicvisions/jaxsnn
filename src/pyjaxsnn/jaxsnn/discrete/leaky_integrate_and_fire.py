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
    dv = dt * params.tau_mem_inv * ((params.v_leak - v) + i)
    v_decayed = v + dv

    # compute current updates
    di = -dt * params.tau_syn_inv * i
    i_decayed = i + di

    # compute new spikes
    z_new = method(v_decayed - params.v_th)
    # compute reset
    v_new = (1 - z_new) * v_decayed + z_new * params.v_reset
    # compute current jumps
    i_new = i_decayed + np.matmul(z, recurrent_weights)
    i_new = i_new + np.matmul(spikes, input_weights)

    return (
        LIFState(z_new, v_new, i_new),
        (input_weights, recurrent_weights),
    ), z_new


def lif_integrate(init, spikes):
    return jax.lax.scan(lif_step, init, spikes)


def LIF(out_dim, scale_in=0.7, scale_rec=0.2):
    """Layer constructor function for a lif (leaky-integrated-fire) layer."""

    def init_fn(rng, input_shape):
        rng, i_key, r_key = random.split(rng, 3)
        input_weights = scale_in * random.normal(i_key, (input_shape, out_dim))
        recurrent_weights = scale_rec * random.normal(
            r_key, (out_dim, out_dim)
        )
        return out_dim, (input_weights, recurrent_weights)

    def apply_fn(weights, inputs, **kwargs):  # pylint: disable=unused-argument
        batch = inputs.shape[1]
        shape = (batch, out_dim)
        state = LIFState(np.zeros(shape), np.zeros(shape), np.zeros(shape))
        (state, _), spikes = jax.lax.scan(lif_step, (state, weights), inputs)

        return spikes

    return init_fn, apply_fn


def LIFStep(
    out_dim, method, scale_in=0.7, scale_rec=0.2, **kwargs
):  # pylint: disable=unused-argument
    """Layer constructor function for a lif (leaky-integrated-fire) layer."""

    def init_fn(rng, input_shape):
        rng, i_key, r_key = random.split(rng, 3)
        input_weights = (
            scale_in * random.normal(i_key, (input_shape, out_dim)) + 0.3
        )
        recurrent_weights = scale_rec * random.normal(
            r_key, (out_dim, out_dim)
        )
        return out_dim, (input_weights, recurrent_weights), rng

    def state_fn(batch_size, **kwargs):  # pylint: disable=unused-argument
        shape = (batch_size, out_dim)
        state = LIFState(np.zeros(shape), np.zeros(shape), np.zeros(shape))
        return state

    lif_step_fn = jax.jit(partial(lif_step, method=method))

    def apply_fn(
        state, weights, inputs, **kwargs
    ):  # pylint: disable=unused-argument
        return lif_step_fn((state, weights), inputs)

    return init_fn, apply_fn, state_fn
