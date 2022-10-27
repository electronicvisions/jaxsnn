from typing import NamedTuple, Tuple

import jax.numpy as np
from jax import jit, random
from jax.lax import scan


class LIState(NamedTuple):
    """State of a leaky-integrator

    Parameters:
        v (np.DeviceArray): membrane voltage
        i (np.DeviceArray): input current
    """

    v: np.DeviceArray
    i: np.DeviceArray


class LIParameters(NamedTuple):
    """Parameters of a leaky integrator

    Parameters:
        tau_syn_inv (np.DeviceArray): inverse synaptic time constant
        tau_mem_inv (np.DeviceArray): inverse membrane time constant
        v_leak (np.DeviceArray): leak potential
    """

    tau_syn_inv: np.DeviceArray = np.array(1.0 / 5e-3)  # type: ignore
    tau_mem_inv: np.DeviceArray = np.array(1.0 / 1e-2)  # type: ignore
    v_leak: np.DeviceArray = np.array(0.0)  # type: ignore


@jit
def li_feed_forward_step(
    init,
    spikes: np.DeviceArray,
    p: LIParameters = LIParameters(),
    dt: float = 0.001,
):
    state, input_weights = init
    # compute current jumps
    i_jump = state.i + np.matmul(spikes, input_weights)
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
) -> Tuple[np.DeviceArray]:
    """Randomly initialize weights for a li layer

    Args:
        input_size (int): input size
        size (int): hidden size
        scale (float, optional): Defaults to 1e-2.

    Returns:
        Tuple[np.DeviceArray]: Randomly initialized weights
    """
    return (scale * random.normal(key, (input_size, size)),)


def li_init_state(size: Tuple[int, int]) -> LIState:
    return LIState(np.zeros(size), np.zeros(size))
