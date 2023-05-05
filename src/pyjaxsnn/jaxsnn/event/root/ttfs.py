import jax.numpy as np

from jaxsnn.functional.leaky_integrate_and_fire import LIFState
from jax import lax


def ttfs_inner_most(a_1, denominator, tau_mem, dt):
    inner_log = 2 * a_1 / denominator
    return lax.cond(
        inner_log > 1,
        lambda: tau_mem * np.log(np.maximum(inner_log, 1)),
        lambda: dt,
    )


def ttfs_inner(a_1, a_2, second_term, tau_mem, dt):
    epsilon = 1e-7
    denominator = a_2 + np.sqrt(np.maximum(second_term, epsilon))
    save_denominator = np.where(np.abs(denominator) > epsilon, denominator, epsilon)
    return lax.cond(
        np.abs(denominator) > epsilon,
        ttfs_inner_most,
        lambda *args: dt,
        a_1,
        save_denominator,
        tau_mem,
        dt,
    )


def ttfs_solver(tau_mem: float, v_th: float, state: LIFState, dt: float):
    """Find the next spike time for special case $\tau_mem = 2 * \tau_syn$

    Args:
        tau_mem float: Membrane time constant
        v_th float: Treshhold Voltage
        state Array: State of the neuron (voltage, current)

    Returns:
        float: Time of next threshhold crossing or dt if no crossing
    """
    v_0, i_0 = state.V, state.I
    a_1 = i_0
    a_2 = v_0 + i_0
    second_term = a_2**2 - 4 * a_1 * v_th
    has_spike = second_term > 0
    return lax.cond(
        has_spike, ttfs_inner, lambda *args: dt, a_1, a_2, second_term, tau_mem, dt
    )
