from functools import partial

import jax
import jax.numpy as np


def solve(f, initial_guess):
    """Newton's method for root-finding.

    Does not need to be differentiable"""

    initial_state = initial_guess

    def body(x, it):
        fx, dfx = f(x), jax.grad(f)(x)
        step = fx / dfx
        return x - step, 0

    res = jax.lax.scan(body, initial_state, np.arange(10))[0]
    return np.where(res > 0, res, np.nan)


def tangent_solve(g, y):
    return y / g(1.0)


def cr_newton_solver(f, y0, initial_guess):
    return jax.lax.custom_root(partial(f, y0), initial_guess, solve, tangent_solve)


if __name__ == "__main__":
    tau_mem = 1e-3
    tau_syn = 5e-4
    tau_mem_inv = 1 / tau_mem
    tau_syn_inv = 1 / tau_syn
    v_th = 0.3
    A = np.array([[-tau_mem_inv, tau_mem_inv], [0, -tau_syn_inv]])
    x0 = np.array([0.0, 2.0])

    def f(x0, t):
        return np.dot(jax.scipy.linalg.expm(A * t), x0)

    def jc(x0, t):
        return f(x0, t)[0] - v_th

    solver = partial(cr_newton_solver, jc)
    print(jax.value_and_grad(solver)(x0, 1e-4))
