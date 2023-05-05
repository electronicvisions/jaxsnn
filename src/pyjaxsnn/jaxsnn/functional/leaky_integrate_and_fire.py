import dataclasses

import jax.numpy as jnp
import tree_math

from jaxsnn.base import explicit
from jaxsnn.base.types import ArrayLike


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

    tau_syn_inv: ArrayLike = 1.0 / 5e-3
    tau_mem_inv: ArrayLike = 1.0 / 1e-2
    v_leak: ArrayLike = 0.0
    v_th: ArrayLike = 0.6
    v_reset: ArrayLike = 0.0

    @property
    def tau_syn(self):
        return 1 / self.tau_syn_inv

    @property
    def tau_mem(self):
        return 1 / self.tau_mem_inv

    @property
    def dynamics(self):
        return jnp.array(
            [[-self.tau_mem_inv, self.tau_mem_inv], [0, -self.tau_syn_inv]]
        )


def lif_dynamics(p: LIFParameters):
    def dynamics(s: LIFState, u: LIFInput):
        v_dot = p.tau_mem_inv * ((p.v_leak - s.V) + s.I + u.I)
        I_dot = -p.tau_syn_inv * s.I
        return LIFState(V=v_dot, I=I_dot)  # , w_rec=0.0)

    return dynamics


def lif_projection(p: LIFParameters, func):  # , rec_fun):
    def projection(state: LIFState, u: LIFInput):
        # TODO: z = func((state.v - p.v_th) / (p.v_th - p.v_leak))
        return LIFState(
            V=jnp.where(state.V > p.v_th, p.v_reset, state.V),
            I=state.I + u.z,  # TODO: + rec_fun(state.w_rec, z)
            # w_rec=state.w_rec,
        )

    return projection


def lif_output(p: LIFParameters, func):
    def output(state: LIFState, _):
        return func((state.V - p.v_th) / (p.v_th - p.v_leak)), state

    return output


def lif_equation(p: LIFParameters, func):
    equation = explicit.ExplicitConstrainedCDE(
        explicit_terms=lif_dynamics(p),
        projection=lif_projection(p, func),
        output=lif_output(p, func),
    )
    return equation
