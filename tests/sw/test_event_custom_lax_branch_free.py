import unittest
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

from jaxsnn.event.modules import (
    LIF,
    Linear,
    Source,
    LIFParameters,
)
from jaxsnn.event.topology import Topology
from jaxsnn.event.types import Spike
from jaxsnn.event.utils import branch_free

from numpy.testing import (
    assert_array_equal,
    assert_array_almost_equal,
)


class TestBranchFreeVsBranched(unittest.TestCase):

    @staticmethod
    def _network():
        params = LIFParameters(
            v_reset=-1000.,
            v_th=1.0,
        )
        builder = Topology(
            t_max=50e-3,
            backprop_method="eventprop",
        )
        builder.add(
            {
                "inp": Source(5),
                "lif": LIF(
                    100,
                    105,
                    params,
                ),
                "syn": Linear(
                    mean=3.0,
                    std=1.6,
                    min_delay=0.000,
                ),
            }
        )
        builder.connect(
            [("inp", "syn"),
             ("syn", "lif")]
        )
        return builder.done()

    @staticmethod
    def _loss_fn(
        apply_fn,
        weights,
        times,
        idxs,
        currents,
        layer_idx,
        internal,
    ):
        input_spikes = Spike(
            times,
            idxs,
            currents,
            layer_idx,
            internal,
        )
        ret = apply_fn(
            {"inp": input_spikes},
            weights,
        )
        return jnp.sum(
            jnp.where(ret["lif"].spike.internal, ret["lif"].spike.time, 0)
        )

    def test_forward_and_grads_match_between_modes(self):
        init_fn, apply_fn = self._network()

        rng = jax.random.PRNGKey(42)
        params = init_fn(rng)

        input_events = Spike(
            time=jnp.sort(np.random.uniform(0, 50e-3, (1, 10))),
            idx=np.random.uniform(0, 5, (1, 10)).astype(int),
            current=jnp.zeros((1, 10)),
            layer_idx=jnp.zeros((1, 10), dtype=int),
            internal=jnp.ones((1, 10), dtype=bool),
        )

        loss_fn = partial(self._loss_fn, apply_fn)

        results = {}
        for branch_free_enabled in (True, False):
            with branch_free(branch_free_enabled):
                forward = apply_fn(
                    {"inp": input_events},
                    params,
                )["lif"].spike
                loss_value, (grad_w, grad_t) = jax.value_and_grad(
                    loss_fn,
                    argnums=(0, 1),
                )(
                    params,
                    input_events.time,
                    input_events.idx,
                    input_events.current,
                    input_events.layer_idx,
                    input_events.internal,
                )
                results[branch_free_enabled] = (
                    forward, loss_value, grad_w, grad_t,
                )

        forward_bf, loss_bf, grad_w_bf, grad_t_bf = results[True]
        forward_br, loss_br, grad_w_br, grad_t_br = results[False]

        self.assertIsNone(assert_array_equal(
            forward_bf.time,
            forward_br.time,
        ))
        self.assertIsNone(assert_array_equal(
            forward_bf.idx,
            forward_br.idx,
        ))
        self.assertIsNone(assert_array_equal(
            forward_bf.internal,
            forward_br.internal,
        ))
        self.assertIsNone(assert_array_equal(
            loss_bf,
            loss_br
        ))
        self.assertIsNone(assert_array_almost_equal(
            grad_w_bf["syn"],
            grad_w_br["syn"],
            6,
        ))
        self.assertIsNone(assert_array_almost_equal(
            grad_t_bf,
            grad_t_br,
            6,
        ))


if __name__ == "__main__":
    unittest.main()
