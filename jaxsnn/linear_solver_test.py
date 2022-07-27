from jaxsnn.linear_solver import hines_solver

import jax.numpy as jnp
import numpy as onp

def test_hines_solver():
    # TODO: This only tests the tri-diagonal case.
    N = 10
    d = 2*onp.random.randn(N)
    u = onp.random.randn(N-1)
    b = onp.random.randn(N)
    p = onp.arange(-1,N,1)
    a = onp.diag(d, 0) + onp.diag(u, 1) + onp.diag(u, -1)

    x = hines_solver(jnp.array(d), jnp.array(u), jnp.array(p),  jnp.array(b))
    x_ = jnp.linalg.solve(a, b)

    # TODO: This is a rather liberal error tolerance...
    onp.testing.assert_allclose(x, x_, rtol=1e-4)