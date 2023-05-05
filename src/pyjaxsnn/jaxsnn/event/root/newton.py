import jax
import jax.numpy as np


# c.f.: https://github.com/google/jax/issues/8744
# def newton_solver(f, initial_guess, state, dt):
#     """Newton's method for root-finding.

#     By using `jax.scan`, this solver is differentiable and jittable.
#     """
#     initial_state = initial_guess

#     def body(x, it):
#         fx, dfx = f(state, x), jax.grad(f, argnums=1)(state, x)
#         step = fx / dfx
#         return x - step, 0

#     res = jax.lax.scan(body, initial_state, np.arange(10))[0]
#     return np.where(res > 0, res, dt)


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
