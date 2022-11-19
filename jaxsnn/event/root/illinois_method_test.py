from .illinois_method import illinois_method

from functools import partial
import jax.numpy as np
import jax


def test_find_spike():
    tau_mem = 1e-3
    tau_syn = 5e-4
    tau_mem_inv = 1 / tau_mem
    tau_syn_inv = 1 / tau_syn
    v_th = 0.3
    A = np.array([[-tau_mem_inv, tau_mem_inv], [0, -tau_syn_inv]])
    x0 = np.array([0.0, 2.0])

    def f(x0, t):
        return np.dot(jax.scipy.linalg.expm(A * t), x0)

    def jc(x0, t):
        return f(x0, t)[0] - v_th

    t = partial(illinois_method, partial(jc, x0))(0.0, 100.0, 0.001)
    print(t)


if __name__ == "__main__":
    test_find_spike()
