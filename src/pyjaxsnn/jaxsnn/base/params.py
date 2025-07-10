import dataclasses
from typing import Dict

import jax
import jax.numpy as jnp
import tree_math


@dataclasses.dataclass
@tree_math.struct
class LIFParameters:
    """
    Parametrization of a LIF neuron

    :param tau_syn: synaptic time constant
    :param tau_mem: membrane time constant
    :param v_th: threshold potential in mV
    :param v_leak: leak potential in mV
    :param v_reset: reset potential in mV
    """

    tau_syn: float = 5e-3
    tau_mem: float = 1e-2
    v_th: float = 0.6
    v_leak: float = 0.0
    v_reset: float = 0.0

    @property
    def dynamics(self) -> jax.Array:
        return jnp.array(
            [[-1. / self.tau_mem, 1. / self.tau_mem], [0, -1. / self.tau_syn]])

    def as_dict(self) -> Dict:
        return {
            "tau_syn": self.tau_syn,
            "tau_mem": self.tau_mem,
            "v_th": self.v_th,
            "v_leak": self.v_leak,
            "v_reset": self.v_reset}


@dataclasses.dataclass
@tree_math.struct
class LIParameters:
    """
    Parametrization of a leaky integrate neuron

    :param tau_syn: synaptic time constant
    :param tau_mem: membrane time constant
    :param v_leak: leak potential
    """

    tau_syn: float = 5e-3
    tau_mem: float = 1e-2
    v_leak: float = 0.0
