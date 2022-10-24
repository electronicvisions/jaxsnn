# Copyright 2022 Christian Pehle
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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




if __name__ == '__main__':
  absltest.main()