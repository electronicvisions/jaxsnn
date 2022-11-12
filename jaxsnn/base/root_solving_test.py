# Copyright (c) 2022 Heidelberg University. All rights reserved.
#
# Released under Apache 2.0 license as described in the file LICENSE.
# Authors: Christian Pehle

from absl.testing import absltest
import numpy as np
import math

import jaxsnn.base.root_solving as root_solving


def test_bisection():
    expected = math.sqrt(2)
    f = lambda x: x**2 - 2

    tol = 0.1
    actual = root_solving.bisection(f, 0, 2, tol)
    np.testing.assert_allclose(expected, actual, atol=tol)

    tol = 0.001
    actual = root_solving.bisection(f, 0, 2, 0.001)
    np.testing.assert_allclose(expected, actual, atol=tol)


def test_newton_1d():
    expected = math.sqrt(2)
    f = lambda x: x**2 - 2
    tol = 0.1
    actual = root_solving.newton_1d(f, 1.0)
    np.testing.assert_allclose(actual, expected, atol=tol)


if __name__ == "__main__":
    absltest.main()
