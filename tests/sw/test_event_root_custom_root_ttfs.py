import unittest

import numpy as np
import jax.numpy as jnp

from jaxsnn.event.solver.custom_root_ttfs import cr_ttfs_solver


class TestEventRootCustomRootTtfs(unittest.TestCase):
    def test_cr_ttfs(self):
        res = cr_ttfs_solver(1e-3, 0.3, jnp.array([0.0, 2.0]), 1e-4)
        self.assertIsNone(
            np.testing.assert_allclose(res, jnp.array(0.000203), atol=1e-06))


if __name__ == '__main__':
    unittest.main()
