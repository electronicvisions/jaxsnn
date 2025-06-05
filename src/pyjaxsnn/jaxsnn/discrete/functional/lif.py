# pylint: disable=invalid-name
from typing import (
    Callable,
    Optional,
    Tuple,
    Dict,
)
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxsnn.base.types import (
    _tm_struct,
    BaseState,
)
from jaxsnn.discrete.types import (
    DenseData,
    Parameter,
)


@dataclass
@_tm_struct
class LIFParameters:
    """
    Parametrization of a LIF neuron

    :param tau_syn: synaptic time constant in s
    :param tau_mem: membrane time constant in s
    :param v_th: threshold potential
    :param v_leak: leak potential
    :param v_reset: reset potential
    """
    tau_syn: float = 5e-3
    tau_mem: float = 1e-2
    v_th: float = 0.6
    v_leak: float = 0.0
    v_reset: float = 0.0


@dataclass
@_tm_struct
class LIFState(BaseState):
    """Leaky-integrate-and-fire state"""
    V: jax.Array
    I: jax.Array  # pylint: disable=disallowed-name


def lif_step(  # pylint: disable=too-many-arguments, too-many-locals
    inputs: Dict[str, DenseData],
    state: LIFState,
    parameters: Optional[Parameter],  # pylint: disable=unused-argument
    method: Callable,
    v_leak: float,
    v_th: float,
    v_reset: float,
    tau_mem: float,
    tau_syn: float,
    dt: float = 0.001,
) -> Tuple[LIFState, DenseData]:
    """Euler step of a leaky-integrate-and-fire neuron.

    :param inputs: Dictionary of input currents
    :param state: Current neuron state
    :param parameters: Optional learnable parameters
    :param method: Surrogate gradient method for the threshold function
    :param v_leak: Leak potential
    :param v_th: Threshold potential
    :param v_reset: Reset potential
    :param tau_mem: Membrane time constant
    :param tau_syn: Synaptic time constant
    :param dt: Time step size

    :return: Tuple of updated neuron state and membrane potential
    """
    inputs_sum = jax.tree_util.tree_reduce(jnp.add, list(inputs.values()))

    dv = dt / tau_mem * ((v_leak - state.V) + state.I)
    v_decayed = state.V + dv

    di = -dt / tau_syn * state.I
    i_decayed = state.I + di

    z_new = method(v_decayed - v_th)
    v_new = (1 - z_new) * v_decayed + z_new * v_reset
    i_new = i_decayed + inputs_sum

    return LIFState(v_new, i_new), z_new
