import jax
import jax.numpy as np

from jaxsnn.types import Array


def ttfs_solver(tau_mem: float, v_th: float, state: Array, dt: float):
    """Find the next spike time for special case $\tau_mem = 2 * \tau_syn$

    Args:
        tau_mem float: Membrane time constant
        v_th float: Treshhold Voltage
        state Array: State of the neuron (voltage, current)

    Returns:
        float: Time of next threshhold crossing or dt if no crossing
    """
    v_0, i_0 = state
    a_1 = i_0
    a_2 = v_0 + i_0
    second_term = a_2**2 - 4 * a_1 * v_th
    has_spike = second_term > 0

    def true_fun():
        inner_log = 2 * a_1 / (a_2 + np.sqrt(second_term))
        return jax.lax.cond(
            inner_log > 1,
            lambda: tau_mem * np.log(inner_log),
            lambda: dt,
        )

    # TODO return a mask if there was an actual spike
    return jax.lax.cond(has_spike, true_fun, lambda: dt)


def batched_ttfs_solver(tau_mem, v_th, state, initial_guess):
    """Find the next spike time for special case $\tau_mem = 2 * \tau_syn$"""
    v_0, i_0 = state[:, 0], state[:, 1]
    a_1 = i_0
    a_2 = v_0 + i_0
    T = tau_mem * np.log(2 * a_1 / (a_2 + np.sqrt(a_2**2 - 4 * a_1 * v_th)))
    return T
