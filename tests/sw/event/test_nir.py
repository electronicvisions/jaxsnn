"""Test the functionality of NIR and NIRData conversions"""
import unittest
import numpy as np
import nir
import jax.numpy as jnp
from jax import random
from jaxsnn import ConversionConfig, from_nir, from_nir_data, to_nir_data
from jaxsnn.base.compose import serial
from jaxsnn.event.modules.leaky_integrate_and_fire import LIF, LIFParameters
from jaxsnn.event.types import EventPropSpike


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

        # training params
        t_max = 4.0 * params.tau_syn
        size = 1
        n_spikes = 10

        jaxsnn_init, jaxsnn_apply = serial(LIF(
            size=size,
            n_spikes=n_spikes,
            t_max=t_max,
            params=params,
            mean=x,
            std=0))

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
        config = ConversionConfig(t_max = 4*params.tau_syn,
                                  n_spikes = {"lif": n_spikes})
        nir_init, nir_apply = from_nir(nir_graph, config)

        input_spikes = EventPropSpike(
            time=jnp.array([0.0, 1e-4, 2e-4, 3e-4, 4e-4]),
            idx=jnp.array([0, 0, 0, 0, 0]),
            current=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]))

        # jaxsnn model without NIR
        _, weights = jaxsnn_init(rng, 1)
        _, _, jaxsnn_output, _ = jaxsnn_apply(weights, input_spikes)

        # jaxsnn model with NIR
        _, nir_weights = nir_init(rng, 1)
        _, _, nir_output, _ = nir_apply(nir_weights, input_spikes)

        # Compare outputs
        assert(jaxsnn_output.idx[5] != -1), "There should be at least one spike in the output."
        assert(
            jnp.array_equal(jaxsnn_output.time, nir_output.time) and
            jnp.array_equal(jaxsnn_output.idx, nir_output.idx) and
            jnp.array_equal(jaxsnn_output.current, nir_output.current)
        ), "NIR to jaxsnn conversion did not produce the same output as the manual jaxsnn model."


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
        cfg = ConversionConfig(t_max=0.02, n_spikes={"lif": 10})
        nir_data = nir.NIRGraphData(
            nodes={
                "lif": nir.NIRNodeData(
                    observables={
                        "spikes": nir.TimeGriddedData(
                            data=np.random.randint(0, 2, (4, 20, 10)).astype(bool),
                            dt=0.001
                        )
                    }
                )
            }
        )
        init, apply = from_nir(self.nir_graph, cfg)
        apply.nodes = cfg.n_spikes
        apply.t_max = cfg.t_max
        jaxsnn_model = (init, apply)
        jaxsnn_dict = from_nir_data(nir_data, jaxsnn_model)

        self.assertIn("lif", jaxsnn_dict,
                      "Converted jaxsnn dict should contain 'lif' node.")
        self.assertIsInstance(jaxsnn_dict["lif"], EventPropSpike,
                              "'lif' node should be of type EventPropSpike.")
        self.assertEqual(jaxsnn_dict["lif"].time.shape,
                         (4, 10),
                         "'lif' spikes should have shape (batch_size, n_spikes).")

    def test_stable_conversion(self):
        cfg = ConversionConfig(t_max=5e-4, n_spikes={"lif": 10})
        init, apply = from_nir(self.nir_graph, cfg)

        original_spikes = {"lif": EventPropSpike(
            time=jnp.array([[0.0, 1e-4, 2.5e-4, 3e-4, 4e-4], [0.0, 1.5e-4, 2.5e-4, 3e-4, 4e-4]]),
            idx=jnp.array([[0, 1, 3, 5, 2], [4, 3, 2, 1, 0]]),
            current=jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]))}

        apply.nodes = cfg.n_spikes
        apply.t_max = cfg.t_max
        jaxsnn_model = (init, apply)
        nir_data = to_nir_data(original_spikes, jaxsnn_model)
        converted_spikes = from_nir_data(nir_data, jaxsnn_model)

        self.assertTrue(jnp.equal(original_spikes["lif"].idx, converted_spikes["lif"].idx).all(),
                        "Mismatch in spike times for node 'lif'")
        self.assertTrue(jnp.equal(original_spikes["lif"].time, converted_spikes["lif"].time).all(),
                        "Mismatch in spike times for node 'lif'")
