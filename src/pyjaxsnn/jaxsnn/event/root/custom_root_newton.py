# pylint: disable=invalid-name
from functools import partial

import jax
import jax.numpy as np


def solve(f, initial_guess, dt):  # pylint: disable=unused-argument
    """Newton's method for root-finding.

    Does not need to be differentiable"""

    initial_state = initial_guess

    def body(x, it):  # pylint: disable=unused-argument
        fx, dfx = f(x), jax.grad(f)(x)
        step = fx / dfx
        return x - step, 0

    res = jax.lax.scan(body, initial_state, np.arange(10))[0]
    return res


def tangent_solve(g, y):
    return y / g(1.0)


def cr_newton_solver(f, initial_guess, state, dt):
    res = jax.lax.custom_root(
        partial(f, state), initial_guess, partial(solve, dt=dt), tangent_solve
    )
    return np.where(res > 0, res, dt)
