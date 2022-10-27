import jax
import jax.numpy as np


def ttfs_solver(tau_mem, v_th, state, initial_guess):
    """Find the next spike time for special case $\tau_mem = 2 * \tau_syn$"""
    v_0, i_0 = state
    a_1 = i_0
    a_2 = v_0 + i_0
    has_spike = a_2**2 - 4 * a_1 * v_th > 0
    true_fun = lambda: tau_mem * np.log(
        2 * a_1 / (a_2 + np.sqrt(a_2**2 - 4 * a_1 * v_th))
    )
    false_fun = lambda: np.nan
    return jax.lax.cond(has_spike, true_fun, false_fun)


def batched_ttfs_solver(tau_mem, v_th, state, initial_guess):
    """Find the next spike time for special case $\tau_mem = 2 * \tau_syn$"""
    v_0, i_0 = state[:, 0], state[:, 1]
    a_1 = i_0
    a_2 = v_0 + i_0
    T = tau_mem * np.log(2 * a_1 / (a_2 + np.sqrt(a_2**2 - 4 * a_1 * v_th)))
    return T
