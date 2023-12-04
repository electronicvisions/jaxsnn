import math

import jaxsnn.base.root_solving as root_solving
import numpy as np
from absl.testing import absltest


def test_bisection():
    expected = math.sqrt(2)

    def f(x):
        return x**2 - 2

    tol = 0.1
    actual = root_solving.bisection(f, 0, 2, tol)
    np.testing.assert_allclose(expected, actual, atol=tol)

    tol = 0.001
    actual = root_solving.bisection(f, 0, 2, 0.001)
    np.testing.assert_allclose(expected, actual, atol=tol)


def test_newton_1d():
    expected = math.sqrt(2)

    def f(x):
        return x**2 - 2

    tol = 0.1
    actual = root_solving.newton_1d(f, 1.0)
    np.testing.assert_allclose(actual, expected, atol=tol)


if __name__ == "__main__":
    absltest.main()
