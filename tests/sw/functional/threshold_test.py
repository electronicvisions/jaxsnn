from absl.testing import absltest
from absl.testing import parameterized

from .threshold import heaviside, superspike, triangular

import numpy as np

HEAVI_TEST_PROBLEMS = [
    dict(testcase_name="below", x=-0.3, expected=0.0),
    dict(testcase_name="above", x=0.3, expected=1.0),
    dict(testcase_name="zero", x=0.0, expected=0.5),
]


class HeavisideTest(parameterized.TestCase):
    @parameterized.named_parameters(HEAVI_TEST_PROBLEMS)
    def test_forward_agrees(self, x, expected):
        np.testing.assert_almost_equal(heaviside(np.full(10, x)), np.full(10, expected))


ALL_TEST_PROBLEMS = [
    dict(testcase_name="super", f=lambda x, alpha: superspike(x, alpha), alpha=80),
    dict(
        testcase_name="triangular", f=lambda x, alpha: triangular(x, alpha), alpha=0.3
    ),
]


class PseudoDerivativeTests(parameterized.TestCase):
    @parameterized.named_parameters(ALL_TEST_PROBLEMS)
    def test_forward_agrees(self, f, alpha):
        x = np.random.randn(100)
        y = f(x, alpha)
        np.testing.assert_array_almost_equal(y, heaviside(x))


if __name__ == "__main__":
    absltest.main()
