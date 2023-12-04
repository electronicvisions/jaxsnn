# pylint: disable=invalid-name
import jax
import jax.numpy as np


def newton_solver(f, initial_guess, state, dt):
    """Newton's method for root-finding."""

    def body(val):
        fx = jax.lax.cond(
            val >= 0.0, lambda: f(state, np.maximum(val, 0.0)), lambda: -val
        )
        # fx = f(state, np.maximum(val, 0.0))
        dfx = jax.grad(f, argnums=1)(state, val)
        step = fx / np.where(np.abs(dfx) > 1e-6, dfx, 1e-6)
        return val - step

    def cond(val):
        return (val >= 0.0) & (val < 1000.0)

    def inner(val, it):
        return jax.lax.cond(cond(val), body, lambda *args: val, val), it

    res, _ = jax.lax.scan(inner, initial_guess, np.arange(6))
    res = np.minimum(np.where((res > 0), res, dt), dt)
    return np.where(f(state, res) + 1e-6 > 0, res, dt)
