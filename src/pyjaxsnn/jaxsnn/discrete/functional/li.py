from typing import Optional, Tuple, Dict
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxsnn.base.types import _tm_struct, BaseState
from jaxsnn.discrete.types import DenseData, Parameter

try:
    from jax.tree import reduce as tree_reduce
except ImportError:
    # for compatibility with jax@:0.4.25
    from jax.tree_util import tree_reduce


@dataclass
@_tm_struct
class LIState(BaseState):  # pylint: disable=invalid-name, disallowed-name
    """Leaky Integrate state - fundamental neuron state"""
    V: jax.Array  # pylint: disable=invalid-name
    I: jax.Array  # pylint: disable=invalid-name, disallowed-name


@dataclass
@_tm_struct
class LIParameters:
    """
    Parametrization of a leaky integrate neuron

    :param tau_syn: Synaptic time constant
    :param tau_mem: Membrane time constant
    :param v_leak: Leak potential
    """
    tau_syn: float = 5e-3
    tau_mem: float = 1e-2
    v_leak: float = 0.0


@jax.jit
def li_step(  # pylint: disable=too-many-arguments
    inputs: Dict[str, DenseData],
    state: LIState,
    parameters: Optional[Parameter],  # pylint: disable=unused-argument
    v_leak: float,
    tau_mem: float,
    tau_syn: float,
    dt: float,  # pylint: disable=invalid-name
) -> Tuple[LIState, DenseData]:
    """Euler step of a leaky integrate neuron.

    :param inputs: Dictionary of input currents
    :param state: Current neuron state
    :param parameters: Optional learnable parameters
    :param v_leak: Leak potential
    :param tau_mem: Membrane time constant
    :param tau_syn: Synaptic time constant
    :param dt: Time step size

    :return: Tuple of updated neuron state and membrane potential
    """
    # sum all inputs
    inputs_sum = tree_reduce(jnp.add, list(inputs.values()))

    i_jump = state.I + inputs_sum
    dv = dt / tau_mem * ((v_leak - state.V) + i_jump)  # pylint: disable=invalid-name
    v_new = state.V + dv

    di = -dt / tau_syn * i_jump  # pylint: disable=invalid-name
    i_decayed = i_jump + di

    return LIState(v_new, i_decayed), v_new
