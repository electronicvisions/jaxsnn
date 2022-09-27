import jax.numpy as jnp
from jax import custom_vjp


@custom_vjp
def heaviside(x):
    return 0.5 + 0.5 * jnp.sign(x)

def heaviside_fwd(x):
    return heaviside(x), (x,)

def heaviside_bwd(res, g):
    (x,) = res
    grad = g / (100.0 * jnp.abs(x) + 1.0) ** 2
    return (grad,)

heaviside.defvjp(heaviside_fwd, heaviside_bwd)
