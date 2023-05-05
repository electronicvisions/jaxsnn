import dataclasses

import jax.numpy as np
import tree_math
from jax import jit, random
from jax.lax import scan

from jaxsnn.base.types import Array, ArrayLike


@dataclasses.dataclass
@tree_math.struct
class LIState:
    """State of a leaky-integrator

    Parameters:
        v (Array): membrane voltage
        i (Array): input current
    """

    v: ArrayLike
    i: ArrayLike


@dataclasses.dataclass
@tree_math.struct
class LIParameters:
    """Parameters of a leaky integrator

    Parameters:
        tau_syn_inv (Array): inverse synaptic time constant
        tau_mem_inv (Array): inverse membrane time constant
        v_leak (Array): leak potential
    """

    tau_syn_inv: ArrayLike = 1.0 / 5e-3
    tau_mem_inv: ArrayLike = 1.0 / 1e-2
    v_leak: ArrayLike = 0.0


@jit
def li_feed_forward_step(
    init,
    spikes: Array,
    p: LIParameters = LIParameters(),
    dt: float = 0.001,
):
    state, input_weights = init
    # compute current jumps
    i_jump = state.i + np.matmul(spikes, input_weights)
    # compute voltage updates
    dv = dt * p.tau_mem_inv * ((p.v_leak - state.v) + i_jump)
    v_new = state.v + dv

    # compute current updates
    di = -dt * p.tau_syn_inv * i_jump
    i_decayed = i_jump + di

    return (LIState(v_new, i_decayed), input_weights), v_new


def li_integrate(init, spikes):
    return scan(li_feed_forward_step, init, spikes)


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
