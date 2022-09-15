from typing import NamedTuple, Tuple

import jax.numpy as jnp
from jax import jit, random
from jax.lax import scan


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

    tau_syn_inv: jnp.DeviceArray = jnp.array(1.0 / 5e-3)  # type: ignore
    tau_mem_inv: jnp.DeviceArray = jnp.array(1.0 / 1e-2)  # type: ignore
    v_leak: jnp.DeviceArray = jnp.array(0.0)  # type: ignore


@jit
def li_feed_forward_step(
    init,
    spikes: jnp.DeviceArray,
    p: LIParameters = LIParameters(),
    dt: float = 0.001,
):
    state, input_weights = init
    # compute current jumps
    i_jump = state.i + jnp.matmul(spikes, input_weights)
    # compute voltage updates
    dv = dt * p.tau_mem_inv * ((p.v_leak - state.v) + i_jump)  # type: ignore
    v_new = state.v + dv

    # compute current updates
    di = -dt * p.tau_syn_inv * i_jump  # type: ignore
    i_decayed = i_jump + di

    return (LIState(v_new, i_decayed), input_weights), v_new


def li_integrate(init, spikes):
    return scan(li_feed_forward_step, init, spikes)


def li_init_weights(
    key: random.KeyArray, input_size: int, size: int, scale: float = 1e-2
) -> Tuple[jnp.DeviceArray]:
    """Randomly initialize weights for a li layer

    Args:
        input_size (int): input size
        size (int): hidden size
        scale (float, optional): Defaults to 1e-2.

    Returns:
        Tuple[jnp.DeviceArray]: Randomly initialized weights
    """
    return (scale * random.normal(key, (input_size, size)),)


def li_init_state(size: Tuple[int, int]) -> LIState:
    return LIState(jnp.zeros(size), jnp.zeros(size))
