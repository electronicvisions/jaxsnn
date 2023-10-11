# pylint: disable=invalid-name
from functools import partial

import jax
import jax.numpy as np


def custom_ttfs(
    tau_mem, v_th, state, f, initial_guess
):  # pylint: disable=unused-argument
    """Find the next spike time for special case $\tau_mem = 2 * \tau_syn$"""
    v_0, i_0 = state
    a_1 = i_0
    a_2 = v_0 + i_0
    T = tau_mem * np.log(2 * a_1 / (a_2 + np.sqrt(a_2**2 - 4 * a_1 * v_th)))
    return T


def tangent_solve(g, y):
    return y / g(1.0)


def cr_ttfs_solver(tau_mem, v_th, y0, initial_guess):
    solve = partial(custom_ttfs, tau_mem, v_th, y0)

    def fn(*args):  # pylint: disable=unused-argument
        return 0.0

    return jax.lax.custom_root(fn, initial_guess, solve, tangent_solve)
