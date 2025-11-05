from dataclasses import replace
from typing import Callable, Union
import jax
import jax.numpy as jnp

from jaxsnn.event.functional.lif.dynamics import exponential_flow
from jaxsnn.event.states import LIFState


def _adjoint_lif_flow_distinct(
    tau_syn: Union[jax.Array, float],
    tau_mem: Union[jax.Array, float],
    v_leak: Union[jax.Array, float],
) -> Callable[[LIFState, jax.Array], LIFState]:
    """ Closed-form adjoint LIF flow for tau_mem != tau_syn """
    rate_diff = 1.0 / tau_mem - 1.0 / tau_syn
    inv_coupling = 1.0 / (tau_syn * rate_diff)

    def flow(state: LIFState, time: jax.Array) -> LIFState:
        decay_m = jnp.exp(-time / tau_mem)
        scaled = jnp.expm1(time * rate_diff)
        decay_s = decay_m * (scaled + 1.0)
        shifted_v = state.V - v_leak
        return replace(
            state,
            V=v_leak + shifted_v * decay_m,
            I=shifted_v * decay_m * scaled * inv_coupling + state.I * decay_s,
        )
    return flow


def _adjoint_lif_flow_equal(
    tau: Union[jax.Array, float],
    v_leak: Union[jax.Array, float],
) -> Callable[[LIFState, jax.Array], LIFState]:
    """ Closed-form adjoint LIF flow for tau_mem == tau_syn """
    def flow(state: LIFState, time: jax.Array) -> LIFState:
        decay = jnp.exp(-time / tau)
        shifted_v = state.V - v_leak
        return replace(
            state,
            V=v_leak + shifted_v * decay,
            I=shifted_v * (time / tau) * decay + state.I * decay,
        )
    return flow


def _adjoint_lif_flow_eigen(
    tau_syn: Union[jax.Array, float],
    tau_mem: Union[jax.Array, float],
    v_leak: Union[jax.Array, float],
) -> Callable[[LIFState, jax.Array], LIFState]:
    """
    Adjoint LIF flow via the generic eigendecomposition-based propagator.

    Only valid for tau_mem != tau_syn.
    """
    eigvals = jnp.array([-1.0 / tau_mem, -1.0 / tau_syn])

    k_adj = 1.0 - (tau_syn / tau_mem)
    eigvecs = jnp.array([[k_adj, 0.0],
                         [1.0, 1.0]])

    eigvecs_inv = (1.0 / k_adj) * jnp.array([[1.0, 0.0],
                                             [-1.0, k_adj]])

    return exponential_flow(v_leak, eigvals, eigvecs, eigvecs_inv)


def adjoint_lif_exponential_flow(
    tau_syn: Union[jax.Array, float],
    tau_mem: Union[jax.Array, float],
    v_leak: Union[jax.Array, float],
    *,
    use_eigendecomposition: bool = False,
) -> Callable[[LIFState, jax.Array], LIFState]:
    """
    Flow of the adjoint LIF dynamics.

    :param use_eigendecomposition: Build the flow from the generic
        eigendecomposition-based propagator instead of the closed form.
        Requires distinct time constants.
    """
    degenerate = bool(jnp.isclose(tau_mem, tau_syn))

    if use_eigendecomposition:
        if degenerate:
            raise ValueError(
                "The eigendecomposition-based adjoint flow is singular for "
                "tau_mem == tau_syn and would yield NaN. It is only available"
                "for distinct time constants."
            )
        return _adjoint_lif_flow_eigen(tau_syn, tau_mem, v_leak)

    if degenerate:
        return _adjoint_lif_flow_equal(tau_mem, v_leak)
    return _adjoint_lif_flow_distinct(tau_syn, tau_mem, v_leak)
