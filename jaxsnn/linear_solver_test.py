from jaxsnn.linear_solver import hines_solver

import jax.numpy as jnp
import numpy as onp

def test_hines_solver():
    N = 10
    d = 2*onp.random.randn(N)
    u = onp.random.randn(N-1)
    b = onp.random.randn(N)
    p = onp.arange(-1,N,1)
    p = onp.arange(-1,N,1)
    a = onp.diag(d, 0) + onp.diag(u, 1) + onp.diag(u, -1)

    x = hines_solver(jnp.array(d), jnp.array(u), jnp.array(b), jnp.array(p))
    x_ = jnp.linalg.solve(a, b)

    assert(onp.allclose(x, x_))

if __name__ == '__main__':
    test_hines_solver()