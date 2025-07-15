"""Test the functionality of from_nir()"""
import unittest
import numpy as np
import nir
import jax.numpy as jnp
from jax import random
from jaxsnn.base.compose import serial
from jaxsnn.event.from_nir import from_nir, ConversionConfig
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
                    tau_mem=np.array([params.tau_mem]),
                    tau_syn=np.array([params.tau_syn]),
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
