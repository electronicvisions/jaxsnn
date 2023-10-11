"""Analytically find the time of the next spike for a LIF neuron for the
special case of $\tau_mem = 2 * \tau_syn$.

When using `jax.vmap` to do this root solving in parallel, `jax.lax.cond`
is mapped to `jax.lax.switch`, meaning that both branches are executed.
Therefore, special care is taken to ensure that no NaNs occur, which would
affect gradient calculation."""

import jax
import jax.numpy as np
from jaxsnn.event.types import LIFState


def ttfs_inner_most(
    a_1: jax.Array, denominator: jax.Array, tau_mem: float, t_max: float
) -> jax.Array:
    inner_log = 2 * a_1 / denominator
    return jax.lax.cond(
        inner_log > 1,
        lambda: tau_mem * np.log(np.maximum(inner_log, 1)),
        lambda: t_max,
    )


def ttfs_inner(
    a_1: jax.Array,
    a_2: jax.Array,
    second_term: jax.Array,
    tau_mem: float,
    t_max: float,
):
    epsilon = 1e-7
    denominator = a_2 + np.sqrt(np.maximum(second_term, epsilon))
    save_denominator = np.where(
        np.abs(denominator) > epsilon, denominator, epsilon
    )
    return jax.lax.cond(
        np.abs(denominator) > epsilon,
        ttfs_inner_most,
        lambda *args: t_max,
        a_1,
        save_denominator,
        tau_mem,
        t_max,
    )


def ttfs_solver(tau_mem: float, v_th: float, state: LIFState, t_max: float):
    """Find the next spike time for special case $\tau_mem = 2 * \tau_syn$

    Args:
        tau_mem float (float): Membrane time constant
        v_th float (float): Treshold Voltage
        state (LIFState): State of the neuron (voltage, current)
        t_max (float): maximum time which is to be searched

    Returns:
        float: Time of next threshhold crossing or t_max if no crossing
    """
    v_0, i_0 = state.V, state.I
    a_1 = i_0
    a_2 = v_0 + i_0
    second_term = a_2**2 - 4 * a_1 * v_th
    has_spike = second_term > 0
    return jax.lax.cond(
        has_spike,
        ttfs_inner,
        lambda *args: t_max,
        a_1,
        a_2,
        second_term,
        tau_mem,
        t_max,
    )
