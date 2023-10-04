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


def exponential_flow(A: jax.Array):
    def flow(x0: jax.Array, t: float):
        return np.dot(linalg.expm(A * t), x0)  # type: ignore

    return lif_wrap(flow)


def lif_exponential_flow(p: LIFParameters):
    A = np.array([[-p.tau_mem_inv, p.tau_mem_inv], [0, -p.tau_syn_inv]])
    return exponential_flow(A)
