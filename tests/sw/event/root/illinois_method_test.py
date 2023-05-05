from .illinois_method import illinois_method

from functools import partial
import jax.numpy as np
import jax
from jaxsnn.event.leaky_integrate import LIFParameters, LIFState

from jax.config import config

config.update("jax_debug_nans", True)


def get_lif_dynamics():
    p = LIFParameters()
    A = np.array([[-p.tau_mem_inv, p.tau_mem_inv], [0, -p.tau_syn_inv]])

    def f(state, t):
        x0 = np.array([state.V, state.I])
        return np.dot(jax.scipy.linalg.expm(A * t), x0)

    def jc(state, t):
        return f(state, t)[0] - p.v_th

    return jc


def test_find_spike():
    state = LIFState(V=0.0, I=2.0)
    jc = get_lif_dynamics()

    t = partial(illinois_method, partial(jc, state))(0.0, 100.0, 0.001)
    print(t)


if __name__ == "__main__":
    test_find_spike()
