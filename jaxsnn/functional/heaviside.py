import jax.numpy as np
from jax import custom_vjp


@custom_vjp
def heaviside(x):
    return 0.5 + 0.5 * np.sign(x)


def heaviside_fwd(x):
    return heaviside(x), (x,)


def heaviside_bwd(res, g):
    (x,) = res
    grad = g / (100.0 * np.abs(x) + 1.0) ** 2
    return (grad,)


heaviside.defvjp(heaviside_fwd, heaviside_bwd)
