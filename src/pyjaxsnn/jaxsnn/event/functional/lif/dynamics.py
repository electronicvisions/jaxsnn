from dataclasses import replace
from typing import Callable, Union
from functools import partial

import jax
import jax.numpy as jnp

from jaxsnn.event.states import LIFState


def lif_wrap(func: Callable) -> Callable[[LIFState, jax.Array], LIFState]:
    def inner(*args):
        res = func(jnp.stack([args[0].V, args[0].I]), *args[1:])
        return replace(args[0], V=res[0], I=res[1])
    return inner


def exponential_flow(
    v_leak: Union[jax.Array, float],
    eigvals: jax.Array,
    eigvecs: jax.Array,
    eigvecs_inv: jax.Array,
) -> Callable[[LIFState, jax.Array], LIFState]:
    def flow(
        v_leak: Union[jax.Array, float],
        initial_state: jax.Array,
        time: jax.Array,
    ) -> jax.Array:
        initial_state = initial_state.at[0].add(-v_leak)

        exp_diag = jnp.exp(time * eigvals)
        E = eigvecs @ (exp_diag[:, None] * eigvecs_inv)
        new_state = E @ initial_state
        new_state = new_state.at[0].add(v_leak)
        new_state = jnp.real(new_state)

        return new_state
    return lif_wrap(partial(flow, v_leak))


def _lif_flow_distinct(
    tau_syn: Union[jax.Array, float],
    tau_mem: Union[jax.Array, float],
    v_leak: Union[jax.Array, float],
) -> Callable[[LIFState, jax.Array], LIFState]:
    """ Closed-form LIF flow for tau_mem != tau_syn """
    rate_diff = 1. / tau_mem - 1. / tau_syn
    inv_coupling = 1. / (tau_mem * rate_diff)

    def flow(state: LIFState, time: jax.Array) -> LIFState:
        decay_m = jnp.exp(-time / tau_mem)
        # exp(t * rate_diff) - 1, so that decay_s == decay_m * exp(t * rate_diff)
        scaled = jnp.expm1(time * rate_diff)
        decay_s = decay_m * (scaled + 1.)
        return replace(
            state,
            V=v_leak + (state.V - v_leak) * decay_m
            + state.I * decay_m * scaled * inv_coupling,
            I=state.I * decay_s,
        )
    return flow


def _lif_flow_equal(
    tau: Union[jax.Array, float],
    v_leak: Union[jax.Array, float],
) -> Callable[[LIFState, jax.Array], LIFState]:
    """ Closed-form LIF flow for tau_mem == tau_syn """
    def flow(state: LIFState, time: jax.Array) -> LIFState:
        decay = jnp.exp(-time / tau)
        return replace(
            state,
            V=v_leak + (state.V - v_leak) * decay
            + state.I * (time / tau) * decay,
            I=state.I * decay,
        )
    return flow


def _lif_flow_eigen(
    tau_syn: Union[jax.Array, float],
    tau_mem: Union[jax.Array, float],
    v_leak: Union[jax.Array, float],
) -> Callable[[LIFState, jax.Array], LIFState]:
    """
    LIF flow via the generic eigendecomposition-based propagator.

    Only valid for tau_mem != tau_syn, since the eigenvector basis below is
    singular for a repeated eigenvalue.
    """
    eigvals = jnp.array([-1. / tau_mem, -1. / tau_syn])
    k = 1. - tau_mem / tau_syn
    eigvecs = jnp.array([[1., 1.],
                         [0., k]])
    eigvecs_inv = (1. / k) * jnp.array([[k, -1.],
                                        [0., 1.]])
    return exponential_flow(v_leak, eigvals, eigvecs, eigvecs_inv)


def lif_exponential_flow(
    tau_syn: Union[jax.Array, float],
    tau_mem: Union[jax.Array, float],
    v_leak: Union[jax.Array, float],
    *,
    use_eigendecomposition: bool = False,
) -> Callable[[LIFState, jax.Array], LIFState]:
    """
    Flow of the LIF dynamics, as a scalar closed form.

    :param use_eigendecomposition: Build the flow from the generic
        eigendecomposition-based propagator instead of the closed form.
        Requires distinct time constants.
    """
    degenerate = bool(jnp.isclose(tau_mem, tau_syn))

    if use_eigendecomposition:
        if degenerate:
            raise ValueError(
                "The eigendecomposition-based flow is singular for "
                "tau_mem == tau_syn (repeated eigenvalue) and would yield "
                "NaN. It is only available for distinct time constants."
            )
        return _lif_flow_eigen(tau_syn, tau_mem, v_leak)

    if degenerate:
        return _lif_flow_equal(tau_mem, v_leak)
    return _lif_flow_distinct(tau_syn, tau_mem, v_leak)
