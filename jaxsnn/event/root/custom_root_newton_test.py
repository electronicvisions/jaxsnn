from . import cr_newton_solver

import jax.numpy as np
import numpy as onp
import jax
from functools import partial

def test_cr_newton_solver():
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

    solver = partial(cr_newton_solver, jc)

    time, grad = jax.value_and_grad(solver)(x0, 1e-4)
    onp.testing.assert_allclose(
        time,
        np.array(0.00020306),
        atol=1e-6    
    )
    onp.testing.assert_allclose(
        grad,
        np.array([-0.00079057, -0.00014528]),
        atol=1e-6
    )
