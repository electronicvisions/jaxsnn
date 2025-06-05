from typing import Callable, Union
import jax
import jax.numpy as jnp
from jax.scipy import linalg

from jaxsnn.event.states import LIFState


def lif_wrap(func: Callable) -> Callable[[LIFState, jax.Array], LIFState]:
    def inner(*args):
        res = func(jnp.stack([args[0].V, args[0].I]), *args[1:])
        args[0].V = res[0]
        args[0].I = res[1]
        return args[0]
    return inner


def exponential_flow(
    kernel: jax.Array,
) -> Callable[[LIFState, jax.Array], LIFState]:
    def flow(initial_state: jax.Array, time: jax.Array) -> jax.Array:
        return jnp.dot(linalg.expm(kernel * time), initial_state)

    return lif_wrap(flow)


def lif_exponential_flow(
    tau_syn: Union[jax.Array, float],
    tau_mem: Union[jax.Array, float],
) -> Callable[[LIFState, jax.Array], LIFState]:
    kernel = jnp.array(
        [[-1. / tau_mem, 1. / tau_mem],
         [0, -1. / tau_syn]])
    return exponential_flow(kernel)
