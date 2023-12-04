import dataclasses
from typing import Dict

import jax
import jax.numpy as np
import tree_math


@dataclasses.dataclass
@tree_math.struct
class LIFParameters:
    """Parametrization of a LIF neuron

    Parameters:
        tau_syn_inv (float): inverse synaptic time constant
        tau_mem_inv (float): inverse membrane time constant
        v_leak (float): leak potential in mV
        v_th (float): threshold potential in mV
        v_reset (float): reset potential in mV
    """

    tau_syn_inv: float = 1.0 / 5e-3
    tau_mem_inv: float = 1.0 / 1e-2
    v_leak: float = 0.0
    v_th: float = 0.6
    v_reset: float = 0.0

    @property
    def tau_syn(self) -> float:
        return 1 / self.tau_syn_inv

    @property
    def tau_mem(self) -> float:
        return 1 / self.tau_mem_inv

    @property
    def dynamics(self) -> jax.Array:
        return np.array(
            [[-self.tau_mem_inv, self.tau_mem_inv], [0, -self.tau_syn_inv]]
        )

    def as_dict(self) -> Dict:
        return {
            "tau_syn": self.tau_syn,
            "tau_mem": self.tau_mem,
            "v_leak": self.v_leak,
            "v_th": self.v_th,
            "v_reset": self.v_reset,
        }


@dataclasses.dataclass
@tree_math.struct
class LIParameters:
    """Parametrization of a leaky integrate neuron

    Parameters:
        tau_syn_inv (float): inverse synaptic time constant
        tau_mem_inv (float): inverse membrane time constant
        v_leak (float): leak potential
    """

    tau_syn_inv: float = 1.0 / 5e-3
    tau_mem_inv: float = 1.0 / 1e-2
    v_leak: float = 0.0
