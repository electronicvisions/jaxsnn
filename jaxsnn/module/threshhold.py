import jax
import jax.numpy as np
from jax import custom_vjp
from jaxsnn.functional.heaviside import heaviside


def Heaviside():
    return heaviside


def SuperSpike(alpha=100.0):
    @custom_vjp
    def step_fn(x):
        return 0.5 + 0.5 * np.sign(x)

    def step_fn_fwd(x):
        return step_fn(x), (x,)

    def step_fn_bwd(res, g):
        (x,) = res
        grad = g / (alpha * np.abs(x) + 1.0) ** 2
        return (grad,)

    step_fn.defvjp(step_fn_fwd, step_fn_bwd)
    return step_fn


def HeaviErfc(k):
    r"""Approximation of the heaviside step function as

    .. math::
        h(x,k) = \frac{1}{2} + \frac{1}{2} \text{erfc}(k x)

    where erfc is the error function.
    """

    @custom_vjp
    def step_fn(x):
        return 0.5 + 0.5 * np.sign(x)

    def step_fn_fwd(x):
        return step_fn(x), (x,)

    def step_fn_bwd(res, g):
        (x,) = res
        derfc = (2 * np.exp(-((k * x) ** 2))) / (np.sqrt(np.pi))

        grad = g * derfc
        return (grad,)

    step_fn.defvjp(step_fn_fwd, step_fn_bwd)
    return step_fn


def HeaviTanh(k):
    r"""Approximation of the heaviside step function as

    .. math::
        h(x,k) = \frac{1}{2} + \frac{1}{2} \text{tanh}(k x)
    """

    @custom_vjp
    def step_fn(x):
        return 0.5 + 0.5 * np.sign(x)

    def step_fn_fwd(x):
        return step_fn(x), (x,)

    def step_fn_bwd(res, g):
        (x,) = res
        dtanh = 1 - np.tanh(x * k) ** 2

        grad = g * dtanh
        return (grad,)

    step_fn.defvjp(step_fn_fwd, step_fn_bwd)
    return step_fn


def Logistic(k):
    r"""Probalistic approximation of the heaviside step function as

    .. math::
        z \sim p(\frac{1}{2} + \frac{1}{2} \text{tanh}(k x))
    """

    @custom_vjp
    def step_fn(x, rng):
        p = 0.5 + 0.5 * np.tanh(k * x)
        return jax.random.bernoulli(rng, p=p)

    def step_fn_fwd(x):
        return step_fn(x), (x,)

    def step_fn_bwd(res, g):
        (x,) = res
        dtanh = 1 - np.tanh(x * k) ** 2

        grad = g * dtanh
        return (grad,)

    step_fn.defvjp(step_fn_fwd, step_fn_bwd)
    return step_fn
