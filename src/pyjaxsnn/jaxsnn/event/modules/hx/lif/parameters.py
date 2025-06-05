from typing import Dict
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jaxsnn.event.types import _tm_struct
from jaxsnn.event.hardware.parameter import HXBaseParameter, HXParameter


@dataclass
@_tm_struct
class NeuronParameters:
    """
    Parametrization of a LIF neuron

    TODO: Merge with jasnn.event.models.lif.parameters.LIFParameters
    """
    tau_syn: HXBaseParameter = HXParameter(10e-6)
    tau_mem: HXBaseParameter = HXParameter(10e-6)
    v_th: HXBaseParameter = HXParameter(125)
    v_leak: HXBaseParameter = HXParameter(80)
    v_reset: HXBaseParameter = HXParameter(80)
    i_synin_gm: HXBaseParameter = HXParameter(500)
    membrane_capacitance: HXBaseParameter = HXParameter(63)
    refractory_time: HXBaseParameter = HXParameter(1e-6)
    synapse_dac_bias: HXBaseParameter = HXParameter(600)
    holdoff_time: HXBaseParameter = HXParameter(0e-6)

    def __post_init__(self):
        for field, value in self.as_dict().items():
            if not isinstance(value, HXBaseParameter):
                setattr(self, field, HXParameter(value))
            if not isinstance(value.hardware_value, jax.Array):
                value.hardware_value = jnp.array(value.hardware_value)
            if not isinstance(value, HXParameter):
                if isinstance(value.model_value, jax.Array):
                    value.model_value = jnp.array(value.model_value)

    @property
    def dynamics(self) -> jax.Array:
        return jnp.array(
            [[-1. / self.tau_mem.model_value, 1. / self.tau_mem.model_value],
             [0, -1. / self.tau_syn.model_value]])

    def as_dict(self) -> Dict[str, HXBaseParameter]:
        return {
            "tau_syn": self.tau_syn,
            "tau_mem": self.tau_mem,
            "threshold": self.v_th,
            "leak": self.v_leak,
            "reset": self.v_reset,
            "i_synin_gm": self.i_synin_gm,
            "membrane_capacitance": self.membrane_capacitance,
            "refractory_time": self.refractory_time,
            "synapse_dac_bias": self.synapse_dac_bias,
            "holdoff_time": self.holdoff_time,
        }
