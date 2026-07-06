import unittest
from functools import partial
import copy

import numpy as np

import jax
import jax.numpy as jnp
from jax import random

from jaxsnn.event.types import Spike
try:
    from jaxsnn.event.hardware.parameter import (
        MixedHXModelParameter,
        HXParameter,
    )
    from jaxsnn.event.modules.hx import (
        HXSource,
        HXLIF,
        HXLinear,
        NeuronParameters,
    )
    hxtorch_available = True
except ImportError:
    hxtorch_available = False
from jaxsnn.event.modules import (
    Source,
    LIF,
    Linear,
    LIFParameters,
)
from jaxsnn.event.topology import Topology

try:
    from jax.tree import map as tree_map
except ImportError:
    # for compatibility with jax@:0.4.25
    from jax.tree_util import tree_map


class TestTopologyHXData(unittest.TestCase):
    tau_mem = 12e-6
    tau_syn = 6e-6
    runtime = 30e-6
    input_size = 25
    hidden_size = 100
    output_size = 3
    batch_size = 2


    @unittest.skipIf(
        not hxtorch_available,
        "Skipping tests because Observables and HXParameters have hxtorch " \
        "dependency and hxtorch is not availab."
    )
    def get_mock_apply_fn(self):
        # SW apply function
        lif_params = LIFParameters(
            v_th=1.0,
            v_leak=0.0,
            v_reset=-1000.0,  # TODO: issue in comparision to hw?
            tau_syn=self.tau_syn,
            tau_mem=self.tau_mem,
        )

        # define topology
        builder = Topology(
            mock=True,
            t_max=self.runtime,
            backprop_method="eventprop",
        )

        # create modules
        builder.add(
            {
                "inp": Source(
                    size=25,
                ),
                "lif1": LIF(
                    size=100,
                    n_steps=130,
                    params=lif_params,
                ),
                "lif2": LIF(
                    size=3,
                    n_steps=103,
                    params=lif_params,
                ),
                "syn1": Linear(
                    mean=0.8,
                    std=0.5,
                    min_delay=0.0,
                ),
                "syn2": Linear(
                    mean=0.3,
                    std=0.5,
                    min_delay=0.0,
                ),
            }
        )

        # connect modules
        builder.connect(
            [
                ("inp", "syn1"),
                ("syn1", "lif1"),
                ("lif1", "syn2"),
                ("syn2", "lif2"),
            ]
        )

        return builder.done()

    def get_hx_apply_fn(self):

        lif_params = NeuronParameters(
            v_th=MixedHXModelParameter(1.0, 150),
            v_leak=MixedHXModelParameter(0.0, 80),
            v_reset=MixedHXModelParameter(0.0, 80),
            refractory_time=HXParameter(30e-6),
            i_synin_gm=HXParameter(500),
            synapse_dac_bias=HXParameter(1000),
            tau_syn=HXParameter(self.tau_syn),
            tau_mem=HXParameter(self.tau_mem),
        )

        # define topology
        builder = Topology(
            mock=False,
            t_max=self.runtime,
            backprop_method="eventprop",
        )

        # create modules
        builder.add(
            {
                "inp": HXSource(
                    size=25,
                    n_events=25,
                ),
                "lif1": HXLIF(
                    size=100,
                    n_steps=130,
                    params=lif_params,
                ),
                "lif2": HXLIF(
                    size=3,
                    n_steps=103,
                    params=lif_params,
                ),
                "syn1": HXLinear(
                    mean=0.8,
                    std=0.5,
                    min_delay=0.0,
                ),
                "syn2": HXLinear(
                    mean=0.3,
                    std=0.5,
                    min_delay=0.0,
                ),
            }
        )

        # connect modules
        builder.connect(
            [
                ("inp", "syn1"),
                ("syn1", "lif1"),
                ("lif1", "syn2"),
                ("syn2", "lif2"),
            ]
        )

        return builder.done()

    def get_inputs(self, rng):
        # TODO: Events at same time
        spike_rng, idx_rng = random.split(rng)
        spike_times = jnp.sort(
            random.uniform(
                spike_rng,
                (1, self.input_size),
                maxval=self.runtime,
            )
        )
        spike_idx = random.randint(
            idx_rng, (1, self.input_size), 0, self.input_size
        )

        return {
            "inp": Spike(
                time=spike_times,
                idx=spike_idx,
                internal=jnp.ones((1, self.input_size), dtype=bool),
                layer_idx=jnp.zeros((1, self.input_size), dtype=jnp.int32),
                current=jnp.zeros((1, self.input_size)),
            )
        }

    def test_custom_vjp_for_hx(self):
        # define trainset and testset
        rng = random.PRNGKey(np.random.randint(0, 10000))
        param_rng, input_rng, rng = random.split(rng, 3)

        # Generate random inputs
        inputs = self.get_inputs(input_rng)

        # Get mock functions and parameters
        mock_init_fn, mock_apply_fn = self.get_mock_apply_fn()
        mock_params = mock_init_fn(param_rng)

        def loss_fn(apply_fn, params, inputs):
            # Simple loss: sum of all spike times in lif2
            spikes = apply_fn(
                inputs,
                params,
            )
            lif2_spikes = spikes["lif2"][spikes["lif2"].internal]
            return jnp.sum(lif2_spikes.time), spikes

        # Compute gradients w.r.t. mock spikes
        (mock_loss, mock_spikes), mock_grads = jax.value_and_grad(
            partial(loss_fn, mock_apply_fn), has_aux=True
        )(
            mock_params, inputs
        )

        mock_spikes = copy.deepcopy(mock_spikes)

        # Define the function that will be returned by the patched
        # _construct_hx_run
        def mocked_hx_run_fn(inputs, params):
            spikes_filtered = {}
            for key, spike in mock_spikes.items():
                if spike is None:
                    continue
                spikes_filtered[key] = spike.get_internal().sort(-1)
            return spikes_filtered

        def mocked_expected_return_type(inputs):
            shapes = {}
            for key, value in mock_spikes.items():
                if value is None:
                    shapes[key] = None
                else:
                    shapes[key] = Spike.empty_like(value)
            return shapes

        # Patch the hardware-specific functions to return our defined function
        with unittest.mock.patch(
            "jaxsnn.event.topology.Topology._construct_hx_run",
            return_value=mocked_hx_run_fn,
        ) as mock_construct, unittest.mock.patch(
            "jaxsnn.event.hardware.experiment.Experiment.expected_return_type",
            side_effect=mocked_expected_return_type,
        ) as mock_return_type:
            _, hx_apply_fn = self.get_hx_apply_fn()

            # Compute gradients w.r.t. params
            (hx_loss, hx_spikes), hx_grads = jax.value_and_grad(
                partial(loss_fn, hx_apply_fn), has_aux=True
            )(
                mock_params, inputs
            )

        self.assertEqual(mock_loss, hx_loss)

        # Spikes should be equal
        self.assertIsNone(
            np.testing.assert_array_equal(
                mock_spikes["lif1"].time,
                hx_spikes["lif1"].time,
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                mock_spikes["lif1"].idx,
                hx_spikes["lif1"].idx,
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                mock_spikes["lif1"].internal,
                hx_spikes["lif1"].internal,
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                mock_spikes["lif1"].layer_idx,
                hx_spikes["lif1"].layer_idx,
            )
        )

        # Grads should be equal
        tree_map(np.testing.assert_array_equal, mock_grads, hx_grads)


if __name__ == '__main__':
    unittest.main()
