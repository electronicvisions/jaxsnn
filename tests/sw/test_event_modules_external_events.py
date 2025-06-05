import unittest
import copy

import numpy as np

import jax
from jax import random
import jax.numpy as jnp

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
    LIFParameters,
    Linear,
)
from jaxsnn.event.topology import Topology



class TestExternalEvents(unittest.TestCase):
    """ Tests the EventProp example implementation """

    tau_mem = 12e-6
    tau_syn = 6e-6
    runtime = 15e-6
    input_size = 5
    hidden_size = 100
    output_size = 3
    duplication = 5
    weight_scale = 56

    @unittest.skipIf(
        not hxtorch_available,
        "Skipping tests because Observables and HXParameters have hxtorch " \
        "dependency and hxtorch is not availab.")
    def test_injecting_existing_event_is_equal(self):
        # SW apply function
        lif_params = LIFParameters(
            v_th=0.6,
            v_leak=0.0,
            v_reset=-1000.0,  # TODO: issue in comparision to hw?
            tau_syn=self.tau_syn,
            tau_mem=self.tau_mem,
        )

        # define trainset and testset
        rng = random.PRNGKey(np.random.randint(0, 10000))
        param_rng, data_rng, rng = random.split(rng, 3)

        spike_times = jnp.sort(
            random.uniform(data_rng, (1, self.input_size), maxval=self.runtime)
        )
        spike_rng, idx_rng = random.split(data_rng)
        spike_times = jnp.sort(
            random.uniform(spike_rng, (1, self.input_size), maxval=self.runtime)
        )
        spike_idx = random.randint(
            idx_rng, (1, self.input_size), 0, self.input_size
        )

        mock_inputs = {
            "inp": Spike(
                time=spike_times,
                idx=spike_idx,
                internal=jnp.ones((1, self.input_size), dtype=bool),
                layer_idx=jnp.zeros((1, self.input_size), dtype=jnp.int32),
                current=jnp.zeros((1, self.input_size)),
            )
        }

        # define topology
        builder = Topology(
            mock=True,
            t_max=50e-6,
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
                    n_steps=125,
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

        mock_init_fn, mock_apply_fn = builder.done()

        # init weights
        mock_params = mock_init_fn(param_rng)

        def loss_fn(params, inputs):
            spikes = mock_apply_fn(
                inputs,
                params,
            )
            lif_spikes = spikes["lif2"].time
            loss = jnp.mean(lif_spikes)
            return loss, spikes

        # Backward
        (mock_loss, mock_spikes), mock_grad = jax.value_and_grad(
            loss_fn,
            has_aux=True
        )(
            mock_params,
            mock_inputs
        )

        hx_inputs = copy.deepcopy(mock_inputs)
        hx_params = copy.deepcopy(mock_params)
        external_spikes = copy.deepcopy(mock_spikes)
        events_h = external_spikes["lif1"].get_internal().sort(-1)
        events_o = external_spikes["lif2"].get_internal().sort(-1)
        external_spikes["lif1"] = events_h[:, :100]
        external_spikes["lif2"] = events_o[:, :3]

        # SW apply function
        lif_params = NeuronParameters(
            v_th=MixedHXModelParameter(0.6, 150),
            v_leak=MixedHXModelParameter(0.0, 80),
            v_reset=MixedHXModelParameter(0.0, 80),  # TODO: issue in comparision to hw?
            refractory_time=HXParameter(30e-6),
            i_synin_gm=HXParameter(500),
            synapse_dac_bias=HXParameter(1000),
            tau_syn=HXParameter(self.tau_syn),
            tau_mem=HXParameter(self.tau_mem),
        )

        # define trainset and testset
        builder = Topology(
            mock=True,
            t_max=50e-6,
            backprop_method="eventprop",
            has_external_events=True,
        )

        # create modules
        builder.add(
            {
                "inp": HXSource(
                    size=25,
                    n_events=0,
                ),
                "lif1": HXLIF(
                    size=100,
                    n_steps=125,
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

        _, hx_apply_fn = builder.done()

        def hx_loss_fn(params, inputs, external_events):
            spikes = hx_apply_fn(
                inputs,
                params,
                external_events,
            )
            lif_spikes = spikes["lif2"].time
            loss = jnp.mean(lif_spikes)
            return loss, spikes

        # Backward
        (hx_loss, hx_spikes), hx_grad = jax.value_and_grad(
            hx_loss_fn,
            has_aux=True,
        )(
            hx_params,
            hx_inputs,
            external_spikes,
        )

        # Loss should be equal
        self.assertEqual(hx_loss, mock_loss)

        # Spikes should be equal
        # NOTE: We dont expect current to be equal (non-zero only at internal)
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

        self.assertIsNone(
            np.testing.assert_array_equal(
                mock_spikes["lif2"].time,
                hx_spikes["lif2"].time,
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                mock_spikes["lif2"].idx,
                hx_spikes["lif2"].idx,
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                mock_spikes["lif2"].internal,
                hx_spikes["lif2"].internal,
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                mock_spikes["lif2"].layer_idx,
                hx_spikes["lif2"].layer_idx,
            )
        )

        # Grads should be equal
        self.assertIsNone(
            np.testing.assert_array_equal(
                mock_grad["syn1"],
                hx_grad["syn1"]
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                mock_grad["syn2"],
                hx_grad["syn2"]
            )
        )


if __name__ == "__main__":
    unittest.main()
