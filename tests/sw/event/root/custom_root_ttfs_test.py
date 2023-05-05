from . import cr_ttfs_solver

import jax.numpy as np
import numpy as onp


def test_cr_ttfs():
    res = cr_ttfs_solver(1e-3, 0.3, np.array([0.0, 2.0]), 1e-4)
    onp.testing.assert_allclose(res, np.array(0.000203), atol=1e-06)
