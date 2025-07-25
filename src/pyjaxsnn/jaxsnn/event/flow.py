import jax
import jax.numpy as jnp
from jax.scipy import linalg
from jaxsnn.base.params import LIFParameters


def lif_wrap(func):
    def inner(*args):
        res = func(jnp.stack([args[0].V, args[0].I]), *args[1:])
        args[0].V = res[0]
        args[0].I = res[1]
        return args[0]

    return inner


def exponential_flow(kernel: jax.Array):
    def flow(initial_state: jax.Array, time: float):
        return jnp.dot(linalg.expm(kernel * time), initial_state)

    return lif_wrap(flow)


def lif_exponential_flow(params: LIFParameters):
    kernel = jnp.array(
        [[-1. / params.tau_mem, 1. / params.tau_mem],
         [0, -1. / params.tau_syn]])
    return exponential_flow(kernel)
