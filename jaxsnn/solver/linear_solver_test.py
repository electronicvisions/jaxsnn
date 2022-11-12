from jaxsnn.solver.linear_solver import hines_solver

import jax.numpy as np
import numpy as onp


def test_hines_solver():
    # TODO: This only tests the tri-diagonal case.
    N = 10
    d = 2 * onp.random.randn(N)
    u = onp.random.randn(N - 1)
    b = onp.random.randn(N)
    p = onp.arange(-1, N, 1)
    a = onp.diag(d, 0) + onp.diag(u, 1) + onp.diag(u, -1)

    x = hines_solver(np.array(d), np.array(u), np.array(p), np.array(b))
    x_ = np.linalg.solve(a, b)

    # TODO: This is a rather liberal error tolerance...
    onp.testing.assert_allclose(x, x_, rtol=1e-4)
