import jax.numpy as jnp
from typing import NamedTuple, Tuple
from jax import random


class LIState(NamedTuple):
    """State of a leaky-integrator

    Parameters:
        v (jnp.DeviceArray): membrane voltage
        i (jnp.DeviceArray): input current
    """

    v: jnp.DeviceArray
    i: jnp.DeviceArray


class LIParameters(NamedTuple):
    """Parameters of a leaky integrator

    Parameters:
        tau_syn_inv (jnp.DeviceArray): inverse synaptic time constant
        tau_mem_inv (jnp.DeviceArray): inverse membrane time constant
        v_leak (jnp.DeviceArray): leak potential
    """

    tau_syn_inv: jnp.DeviceArray = jnp.array(1.0 / 5e-3)
    tau_mem_inv: jnp.DeviceArray = jnp.array(1.0 / 1e-2)
    v_leak: jnp.DeviceArray = jnp.array(0.0)


def li_feed_forward_step(
    state: LIState,
    input_weights: jnp.DeviceArray,
    spikes: jnp.DeviceArray,
    p: LIParameters = LIParameters(),
    dt: float = 0.001,
) -> Tuple[jnp.DeviceArray, LIState]:
    # compute current jumps
    i_jump = state.i + jnp.einsum("s,ns->n", spikes, input_weights)
    # compute voltage updates
    dv = dt * p.tau_mem_inv * ((p.v_leak - state.v) + i_jump)
    v_new = state.v + dv

    # compute current updates
    di = -dt * p.tau_syn_inv * i_jump
    i_decayed = i_jump + di

    return LIState(v_new, i_decayed), v_new


def li_init_weights(key: random.KeyArray, input_size: float, size: float, scale: float = 1e-2) -> Tuple[jnp.DeviceArray]:
    """Randomly initialize weights for a li layer

    Args:
        input_size (int): input size
        size (int): hidden size
        scale (float, optional): Defaults to 1e-2.

    Returns:
        Tuple[jnp.DeviceArray]: Randomly initialized weights
    """
    return (scale * random.normal(key, (size, input_size)),)


def li_init_state(size: int) -> LIState:
    return LIState(jnp.zeros(size), jnp.zeros(size))
