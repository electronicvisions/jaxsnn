import dataclasses

import jax.numpy as jnp
import tree_math
from jaxsnn.base.types import Array, ArrayLike


@dataclasses.dataclass
@tree_math.struct
class LIFState:
    """State of a LIF neuron

    Parameters:
        V (ArrayLike): membrane potential
        I (ArrayLike): synaptic input current
    """

    V: ArrayLike
    I: ArrayLike


@dataclasses.dataclass
@tree_math.struct
class LIFInput:
    """Input to a LIF neuron

    Parameters:
        I (ArrayLike): membrane input current
        z (ArrayLike): input spikes
    """

    I: ArrayLike
    z: ArrayLike


@dataclasses.dataclass
@tree_math.struct
class LIFParameters:
    """Parametrization of a LIF neuron

    Parameters:
        tau_syn_inv (ArrayLike): inverse synaptic time
                                    constant (:math:`1/\\tau_\\text{syn}`) in 1/ms
        tau_mem_inv (ArrayLike): inverse membrane time
                                    constant (:math:`1/\\tau_\\text{mem}`) in 1/ms
        v_leak (ArrayLike): leak potential in mV
        v_th (ArrayLike): threshold potential in mV
        v_reset (ArrayLike): reset potential in mV
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
    def dynamics(self) -> Array:
        return jnp.array(
            [[-self.tau_mem_inv, self.tau_mem_inv], [0, -self.tau_syn_inv]]
        )


def lif_dynamics(p: LIFParameters):
    def dynamics(s: LIFState, u: LIFInput):
        v_dot = p.tau_mem_inv * ((p.v_leak - s.V) + s.I + u.I)
        I_dot = -p.tau_syn_inv * s.I
        return LIFState(V=v_dot, I=I_dot)  # , w_rec=0.0)

    return dynamics
