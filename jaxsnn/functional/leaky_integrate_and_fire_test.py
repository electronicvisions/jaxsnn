import jaxsnn.base.explicit as explicit
import jaxsnn.base.funcutils as funcutils
import jaxsnn.functional.threshold as threshold
import jaxsnn.functional.leaky_integrate_and_fire as lif

import jax
import numpy as np

from absl.testing import absltest


class ForwardTests(absltest.TestCase):
    def test_leak_over_threshold(self):
        p = lif.LIFParameters(v_leak=0.6, v_th=0.5)
        initial_state = lif.LIFState(v=p.v_reset, I=0.0, w_rec=0.0)

        T = 1000
        dt = 0.0001
        step_fn = explicit.classic_rk4_cde(
            lif.lif_equation(p, threshold.triangular), dt
        )

        stim = lif.LIFInput(I=np.zeros(T), z=np.zeros(T))
        integrator = funcutils.controlled_trajectory(step_fn, stim)
        integrator = jax.jit(integrator)

        _, (spikes, _) = integrator(initial_state)

        # NOTE: need to use (1-spikes), because output for leak > threshold is inverted
        np.testing.assert_equal(np.sum(1 - spikes), np.array(5.0))


if __name__ == "__main__":
    absltest.main()
