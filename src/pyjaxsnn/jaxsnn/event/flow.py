import jax
import jax.numpy as np
from jax.scipy import linalg
from jaxsnn.base.params import LIFParameters


def lif_wrap(func):
    def inner(*args):
        res = func(np.stack([args[0].V, args[0].I]), *args[1:])
        args[0].V = res[0]
        args[0].I = res[1]
        return args[0]

    return inner


def exponential_flow(kernel: jax.Array):
    def flow(initial_state: jax.Array, time: float):
        return np.dot(linalg.expm(kernel * time), initial_state)

    return lif_wrap(flow)


def lif_exponential_flow(params: LIFParameters):
    kernel = np.array(
        [[-params.tau_mem_inv, params.tau_mem_inv], [0, -params.tau_syn_inv]]
    )
    return exponential_flow(kernel)
