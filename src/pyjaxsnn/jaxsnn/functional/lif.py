from functools import partial
from typing import NamedTuple

import jax.numpy as np
from jax import jit, random
from jax.lax import scan

from jaxsnn.base.types import Array
from jaxsnn.functional.leaky_integrate_and_fire import LIFParameters
from jaxsnn.functional.threshold import superspike


class LIFState(NamedTuple):
    """State of a LIF neuron

    Parameters:
        z (Array): recurrent spikes
        v (Array): membrane potential
        i (Array): synaptic input current
        input_weights (Array): input weights
        recurrent_weights (Array): recurrentweights
    """

    z: Array
    v: Array
    i: Array


def lif_step(
    init,
    spikes,
    method=superspike,
    params: LIFParameters = LIFParameters(),
    dt=0.001,
):
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

    return (LIFState(z_new, v_new, i_new), (input_weights, recurrent_weights)), z_new


def lif_integrate(init, spikes):
    return scan(lif_step, init, spikes)


def LIF(out_dim, scale_in=0.7, scale_rec=0.2):
    """Layer constructor function for a lif (leaky-integrated-fire) layer."""

    def init_fn(rng, input_shape):
        rng, i_key, r_key = random.split(rng, 3)
        input_weights = scale_in * random.normal(i_key, (input_shape, out_dim))
        recurrent_weights = scale_rec * random.normal(r_key, (out_dim, out_dim))
        return out_dim, (input_weights, recurrent_weights)

    def apply_fn(params, inputs, **kwargs):
        batch = inputs.shape[1]
        shape = (batch, out_dim)
        state = LIFState(np.zeros(shape), np.zeros(shape), np.zeros(shape))
        (state, _), spikes = scan(lif_step, (state, params), inputs)

        return spikes

    return init_fn, apply_fn


def LIFStep(out_dim, method, scale_in=0.7, scale_rec=0.2, **kwargs):
    """Layer constructor function for a lif (leaky-integrated-fire) layer."""

    def init_fn(rng, input_shape):
        rng, i_key, r_key = random.split(rng, 3)
        input_weights = scale_in * random.normal(i_key, (input_shape, out_dim))
        recurrent_weights = scale_rec * random.normal(r_key, (out_dim, out_dim))
        return out_dim, (input_weights, recurrent_weights), rng

    def state_fn(batch_size, **kwargs):
        shape = (batch_size, out_dim)
        state = LIFState(np.zeros(shape), np.zeros(shape), np.zeros(shape))
        return state

    lif_step_fn = jit(partial(lif_step, method=method))

    def apply_fn(state, params, inputs, **kwargs):
        return lif_step_fn((state, params), inputs)

    return init_fn, apply_fn, state_fn
