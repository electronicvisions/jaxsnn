# pylint: disable=invalid-name
import jax
import jax.numpy as np


def linear_interpolation(f_a, f_b, a, b, x):
    return (x - a) / (b - a) * f_b + (b - x) / (b - a) * f_a


def linear_interpolated_root(f_a, f_b, a, b):
    return (a * f_b - b * f_a) / f_b - f_a


def newton_1d(f, x0):
    initial_state = (0, x0)

    def cond(state):
        it, _ = state
        return it < 10

    def body(state):
        it, x = state
        fx, dfx = f(x), jax.grad(f)(x)
        step = fx / dfx
        new_state = it + 1, x - step
        return new_state

    return jax.lax.while_loop(
        cond,
        body,
        initial_state,
    )[1]


def newton_nd(f, x0):
    initial_state = (0, x0)

    def cond(state):
        it, _ = state
        return it < 10

    def body(state):
        it, x = state
        fx, dfx = f(x), jax.grad(f)(x)
        step = jax.numpy.linalg.solve(dfx, -fx)

        new_state = it + 1, x + step
        return new_state

    return jax.lax.while_loop(
        cond,
        body,
        initial_state,
    )[1]


def bisection(f, x_min, x_max, tol):
    """Bisection root finding method

    Based on the intermediate value theorem, which
    guarantees for a continuous function that there
    is a zero in the interval [x_min, x_max] as long
    as sign(f(x_min)) != sign(f(x_max)).

    NOTE: We do not check the precondition sign(f(x_min)) != sign(f(x_max))
    """
    initial_state = (0, x_min, x_max)  # (iteration, x)

    def cond(state):
        _, x_min, _ = state
        return np.abs(f(x_min)) > tol  # it > 10

    def body(state):
        it, x_min, x_max = state
        x = (x_min + x_max) / 2

        sfxm = np.sign(f(x_min))
        sfx = np.sign(f(x))

        x_min = np.where(sfx == sfxm, x, x_min)
        x_max = np.where(sfx == sfxm, x_max, x)

        new_state = (it + 1, x_min, x_max)
        return new_state

    return jax.lax.while_loop(
        cond,
        body,
        initial_state,
    )[1]
