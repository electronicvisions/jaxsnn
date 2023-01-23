import jax
import jax.numpy as np


# c.f.: https://github.com/google/jax/issues/8744
def newton_solver(f, initial_guess, state, dt):
    """Newton's method for root-finding.

    By using `jax.scan`, this solver is differentiable and jittable.
    """
    initial_state = initial_guess

    def body(x, it):
        fx, dfx = f(state, x), jax.grad(f, argnums=1)(state, x)
        step = fx / dfx
        return x - step, 0

    res = jax.lax.scan(body, initial_state, np.arange(10))[0]
    return np.where(res > 0, res, dt)
