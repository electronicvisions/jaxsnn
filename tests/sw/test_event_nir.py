"""Test the functionality of NIR and NIRData conversions"""
import unittest
import numpy as np

import jax.numpy as jnp
from jax import random

from jaxsnn.event.modules import (
    LIF,
    LIFParameters,
    Linear,
    Source,
)
from jaxsnn.event.topology import Topology
from jaxsnn.event.types import Spike

import pytest
nir = pytest.importorskip("nir")

try:
    from jaxsnn.event import (
        ConversionConfig,
        from_nir,
        from_nir_data,
        to_nir_data,
    )
except ImportError:
    pass


class TestFromNIRConversion(unittest.TestCase):

    def test_compare_nir_to_no_nir(self):
        """
        Test that a jaxsnn model converted from NIR produces the same output
        as a manually constructed jaxsnn model without NIR.
        """

        rng = random.PRNGKey(1234)
        # generate random number x as weight
        x = float(random.uniform(rng, shape=(), minval=2.0, maxval=3.0))
        params = LIFParameters(v_reset=-1.0, v_th=1.0,
                               tau_mem=1e-2, tau_syn=5e-3)

        builder = Topology(t_max=4.0 * params.tau_syn)

        builder.add(
            {
                "input": Source(1),
                "lif": LIF(
                    1,
                    10,
                    params,
                ),
                "syn1": Linear(
                    mean=x,
                    std=0.,
                    min_delay=0.000,
                ),
            }
        )

        builder.connect(
            [
                ("input", "syn1"),
                ("syn1", "lif")
            ]
        )

        params = LIFParameters(v_reset=-1.0, v_th=1.0,
                               tau_mem=1e-2, tau_syn=5e-3)

        # Create a NIR graph that matches the LIF model and convert it
        # to jaxsnn init/apply functions
        nir_graph = nir.NIRGraph(
            nodes={
                "input": nir.Input(input_type=np.array([1])),
                "linear": nir.Linear(weight=np.array([[x]])),
                "lif": nir.CubaLIF(
                    tau_mem=np.array([params.tau_mem * 1e3]),
                    tau_syn=np.array([params.tau_syn * 1e3]),
                    r=np.array([1]),
                    v_leak=np.array([params.v_leak]),
                    v_reset=np.array([params.v_reset]),
                    v_threshold=np.array([params.v_th])
                ),
                "output": nir.Output(output_type=np.array([1]))
            },
            edges=[
                ("input", "linear"),
                ("linear", "lif"),
                ("lif", "output")
            ]
        )

        jaxsnn_init, jaxsnn_apply = builder.done()

        config = ConversionConfig(
            t_max=4 * params.tau_syn,
            n_steps={"lif": 10},
        )
        nir_topology = from_nir(nir_graph, config)
        nir_init, nir_apply = nir_topology.done()

        input_spikes = {
            "input": Spike(
                time=jnp.array([[0.0, 1e-4, 2e-4, 3e-4, 4e-4]]),
                idx=jnp.array([[0, 0, 0, 0, 0]]),
                current=jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0]]),
                layer_idx=jnp.array([[0, 0, 0, 0, 0]]),
                internal=jnp.array([[True, True, True, True, True]]),
            )
        }

        # jaxsnn model without NIR
        weights = jaxsnn_init(rng)
        jaxsnn_output = jaxsnn_apply(input_spikes, weights)

        # jaxsnn model with NIR
        nir_weights = nir_init(rng)
        nir_output = nir_apply(input_spikes, nir_weights)

        # Compare outputs
        self.assertTrue(
            jaxsnn_output['lif'].idx[0, 5] != -1,
            "There should be at least one spike in the output.",
        )

        self.assertTrue(
            jnp.allclose(
                jaxsnn_output['lif'].time, nir_output['lif'].time,
            )
            and jnp.array_equal(
                jaxsnn_output['lif'].idx, nir_output['lif'].idx,
            )
            and jnp.allclose(
                jaxsnn_output['lif'].current, nir_output['lif'].current,
            )
            and jnp.array_equal(
                jaxsnn_output['lif'].layer_idx, nir_output['lif'].layer_idx,
            )
            and jnp.array_equal(
                jaxsnn_output['lif'].internal, nir_output['lif'].internal,
            ),
            "NIR to jaxsnn conversion did not produce the same output as the "
            "manual jaxsnn model."
        )


class TestNIRDataConversion(unittest.TestCase):

    nir_graph = nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type=np.array([5])),
            "linear": nir.Linear(weight=np.random.rand(10, 5)),
            "lif": nir.CubaLIF(
                tau_mem=np.array([0.02] * 10),
                tau_syn=np.array([0.005] * 10),
                r=np.array([1.0] * 10),
                v_leak=np.array([0.1] * 10),
                v_reset=np.array([0.0] * 10),
                v_threshold=np.array([1.0] * 10)
            ),
            "output": nir.Output(output_type=np.array([10]))
        },
        edges=[
            ("input", "linear"),
            ("linear", "lif"),
            ("lif", "output")
        ]
    )

    def test_from_time_gridded_data(self):
        cfg = ConversionConfig(t_max=0.02, n_steps={"lif": 10})
        nir_data = nir.NIRGraphData(
            nodes={
                "lif": nir.NIRNodeData(
                    observables={
                        "spikes": nir.TimeGriddedData(
                            data=np.random.randint(
                                0, 2, (4, 20, 10),
                            ).astype(bool),
                            dt=0.001
                        )
                    }
                )
            }
        )
        topology = from_nir(self.nir_graph, cfg)

        jaxsnn_dict = from_nir_data(nir_data, topology)

        self.assertIn(
            "lif",
            jaxsnn_dict,
            "Converted jaxsnn dict should contain 'lif' node.",
        )
        self.assertIsInstance(
            jaxsnn_dict["lif"],
            Spike,
            "'lif' node should be of type Spike.",
        )
        self.assertEqual(
            jaxsnn_dict["lif"].time.shape,
            (4, 10),
            "'lif' spikes should have shape (batch_size, n_spikes).",
        )

    def test_stable_conversion(self):
        cfg = ConversionConfig(t_max=5e-4, n_steps={"lif": 10})
        topology = from_nir(self.nir_graph, cfg)

        original_spikes = {
            "lif": Spike(
                time=jnp.array([[0.0, 1e-4, 2.5e-4, 3e-4, 4e-4],
                                [0.0, 1.5e-4, 2.5e-4, 3e-4, 4e-4]]),
                idx=jnp.array([[0, 1, 3, 5, 2],
                               [4, 3, 2, 1, 0]]),
                current=jnp.array([[0.0, 3.0, 4.5, 0.0, 2.7],
                                   [0.5, 0.9, 2.0, 2.6, 1.9]]),
                layer_idx=jnp.array([[0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0]]),
                internal=jnp.array([[True, True, True, True, True],
                                    [True, True, True, True, True]])
        )}

        nir_data = to_nir_data(
            original_spikes,
            topology,
            ('spikes', 'current'),
        )
        converted_spikes = from_nir_data(
            nir_data,
            topology,
            ('spikes', 'current'),
        )

        self.assertTrue(
            jnp.array_equal(
                original_spikes["lif"].idx,
                converted_spikes["lif"].idx,
            ),
            "Mismatch in spike indices for node 'lif'",
        )
        self.assertTrue(
            jnp.allclose(
                original_spikes["lif"].time,
                converted_spikes["lif"].time,
            ),
            "Mismatch in spike times for node 'lif'",
        )
        self.assertTrue(
            jnp.allclose(
                original_spikes["lif"].current,
                converted_spikes["lif"].current,
            ),
            "Mismatch in current for node 'lif'",
        )
        self.assertTrue(
            jnp.array_equal(
                original_spikes["lif"].layer_idx,
                converted_spikes["lif"].layer_idx,
            ),
            "Mismatch in layer_idx for node 'lif'",
        )
        self.assertTrue(
            jnp.array_equal(
                original_spikes["lif"].internal,
                converted_spikes["lif"].internal,
            ),
            "Mismatch in internal for node 'lif'",
        )


if __name__ == "__main__":
    unittest.main()
