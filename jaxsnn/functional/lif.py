from typing import NamedTuple, Tuple

import jax.numpy as np
from jax import random
from jax.lax import scan

from jaxsnn.functional.heaviside import heaviside
from jaxsnn.types import Array


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


class LIFParameters(NamedTuple):
    """Parametrization of a LIF neuron

    Parameters:
        tau_syn_inv (Array): inverse synaptic time
                                    constant (:math:`1/\\tau_\\text{syn}`) in 1/ms
        tau_mem_inv (Array): inverse membrane time
                                    constant (:math:`1/\\tau_\\text{mem}`) in 1/ms
        v_leak (Array): leak potential in mV
        v_th (Array): threshold potential in mV
        v_reset (Array): reset potential in mV
        method (str): method to determine the spike threshold
                      (relevant for surrogate gradients)
        alpha (float): hyper parameter to use in surrogate gradient computation
    """

    tau_syn_inv: Array = np.array(1.0 / 5e-3)
    tau_mem_inv: Array = np.array(1.0 / 1e-2)
    v_leak: Array = np.array(0.0)
    v_th: Array = np.array(0.5)
    v_reset: Array = np.array(0.0)


def lif_current_encoder(
    voltage,
    input_current: Array,
    p: LIFParameters = LIFParameters(),
    dt: float = 0.001,
) -> Tuple[Array, Array]:
    r"""Computes a single euler-integration step of a leaky integrator. More
    specifically it implements one integration step of the following ODE

    .. math::
        \begin{align*}
            \dot{v} &= 1/\tau_{\text{mem}} (v_{\text{leak}} - v + i) \\
            \dot{i} &= -1/\tau_{\text{syn}} i
        \end{align*}

    Parameters:
        input (Array): the input current at the current time step
        voltage (Array): current state of the LIF neuron
        p (LIFParameters): parameters of a leaky integrate and fire neuron
        dt (float): Integration timestep to use
    """
    dv = dt * p.tau_mem_inv * ((p.v_leak - voltage) + input_current)
    voltage = voltage + dv
    z = heaviside(voltage - p.v_th)

    voltage = voltage - z * (voltage - p.v_reset)
    return voltage, z


def lif_step(
    init,
    spikes,
    method=heaviside,
    params: LIFParameters = LIFParameters(),
    dt=0.001,
):
    state, weights = init
    input_weights, recurrent_weights = weights
    z, v, i = state
    tau_syn_inv, tau_mem_inv, v_leak, v_th, v_reset = params

    # compute voltage updates
    dv = dt * tau_mem_inv * ((v_leak - v) + i)
    v_decayed = v + dv

    # compute current updates
    di = -dt * tau_syn_inv * i
    i_decayed = i + di

    # compute new spikes
    z_new = method(v_decayed - v_th)
    # compute reset
    v_new = (1 - z_new) * v_decayed + z_new * v_reset
    # compute current jumps
    i_new = i_decayed + np.matmul(z, recurrent_weights)
    i_new = i_new + np.matmul(spikes, input_weights)

    return (LIFState(z_new, v_new, i_new), (input_weights, recurrent_weights)), z_new


def liv_derivative(t, y, **kwargs):
    v, i = y
    tau_syn_inv, tau_mem_inv, v_leak, v_th, v_reset = LIFParameters()

    dv = tau_mem_inv * ((v_leak - v) + i)
    di = -tau_syn_inv * i
    return (dv, di)


def lif_integrate(init, spikes):
    return scan(lif_step, init, spikes)


def lif_init_weights(
    key: random.KeyArray, input_size: int, size: int, scale: float = 1e-2
) -> Tuple[Array, Array]:
    """Randomly initialize weights and recurrent weights for a snn layer

    Args:
        input_size (int): input size
        size (int): hidden size
        scale (float, optional): Defaults to 1e-2.

    Returns:
        Tuple[Array, Array]: Randomly initialized weights
    """
    i_key, r_key = random.split(key)
    input_weights = scale * random.normal(i_key, (input_size, size))
    recurrent_weights = scale * random.normal(r_key, (size, size))
    return input_weights, recurrent_weights


def lif_init_state(size: Tuple[int, int]) -> LIFState:
    return LIFState(np.zeros(size), np.zeros(size), np.zeros(size))
