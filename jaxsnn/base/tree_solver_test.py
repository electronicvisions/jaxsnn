# Copyright (c) 2022 Heidelberg University. All rights reserved.
#
# Released under Apache 2.0 license as described in the file LICENSE.
# Authors: Christian Pehle

from jaxsnn.base.tree_solver import hines_solver, tree_to_matrix

import jax.numpy as jnp
import numpy as onp

from absl.testing import absltest


def test_hines_solver():
    N = 10
    d = 2 * onp.random.randn(N)
    u = onp.random.randn(N - 1)
    b = onp.random.randn(N)
    p = onp.arange(-1, N, 1)
    a = onp.diag(d, 0) + onp.diag(u, 1) + onp.diag(u, -1)

    x = hines_solver(jnp.array(d), jnp.array(u), jnp.array(p), jnp.array(b))
    x_ = jnp.linalg.solve(a, b)

    # TODO: This is a rather liberal error tolerance...
    onp.testing.assert_allclose(x, x_, rtol=1e-4)


def test_tree_to_matrix():
    N = 3
    d = 2 * onp.ones(N)
    u = onp.ones(N - 1)
    p = onp.arange(-1, N, 1)

    expected = [[2, 1, 0], [1, 2, 1], [0, 1, 2]]
    actual = tree_to_matrix(d, u, p)
    onp.testing.assert_allclose(expected, actual)


if __name__ == "__main__":
    absltest.main()
