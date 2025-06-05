"""Conversion of a NIR graph to jaxsnn-model"""
from dataclasses import dataclass
from typing import Dict, Union

import numpy as np
import jax
import jax.numpy as jnp

from jaxsnn.base.types import Parameter
from jaxsnn.event.types import Projection, Population, SourcePopulation
from jaxsnn.event.modules import Source, LIF, Linear
from jaxsnn.event.modules.lif import LIFParameters
from jaxsnn.event.topology import Topology
import nir


@dataclass
class ConversionConfig:
    """
    Configuration for the conversion from NIR to jaxsnn.

    :param t_max: Maximum time for the simulation.
    :param n_steps: Dictionary with number of spikes to simulate per layer.
    :param backprop_method: Either "eventprop" for the EventProp algorithm or
        "analytical" for the time-to-first-spike solver.
    """
    t_max: float
    n_steps: Dict[str, int]
    backprop_method: str = "analytical"


def node_from_nir(node_key: str, node: nir.NIRNode, config: ConversionConfig
                  ) -> Union[Population, Projection, SourcePopulation]:
    """
    Convert a NIR node to a jaxsnn module.

    :param node_key: Key of the node in the NIR graph.
    :param node: NIR node to convert.
    :param config: Conversion configuration.

    :return: Converted jaxsnn module.
    """
    if isinstance(node, nir.Input):
        return Source(node.input_type['input'])

    if isinstance(node, nir.CubaLIF):
        # assert that n_steps is defined for every CubaLIF neuron
        if node_key not in config.n_steps:
            raise KeyError(
                f"Number of event steps in the simulation for {node_key} not \
                defined in config.n_steps")
        size = np.size(node.tau_mem)
        params = LIFParameters(
            v_leak=node.v_leak[0],
            v_reset=node.v_reset[0],
            v_th=node.v_threshold[0],
            tau_mem=node.tau_mem[0] * 1e-3,
            tau_syn=node.tau_syn[0] * 1e-3,
        )
        return LIF(size=size, n_steps=config.n_steps[node_key], params=params)

    if isinstance(node, nir.Linear):
        linear = Linear(mean=0.0, std=0.0)

        def generator(
            input_size: int,  # pylint: disable=unused-argument
            output_size: int,  # pylint: disable=unused-argument
        ) -> Projection.Functions:
            """ """
            def init_fn(rng: jax.Array) -> Parameter:  # pylint: disable=unused-argument
                return jnp.asarray(node.weight)

            def state_fn(*args) -> None:  # pylint: disable=unused-argument
                return None

            def event_fn(*args) -> None:  # pylint: disable=unused-argument
                return None

            return Projection.Functions(init_fn, state_fn, event_fn)
        linear.generator = generator
        return linear

    raise NotImplementedError(f"Node type {type(node)} not supported yet")


def from_nir(graph: nir.NIRGraph, config: ConversionConfig):
    """
    Convert NIRGraph to jax-snn Topology

    Restrictions for NIRGraph:
    - Only linear feed-forward SNNs are supported
    - CubaLIF and Linear layers are supported
    - Affine layers with bias==0 are currently supported
    - In terms of parameters, only homogeneous layers are supported
    - The analytical solver is only supported for non-external inputs

    :param graph: NIR graph to convert
    :param config: Conversion configuration

    Example:
    ```python
    nir_graph = nir.NIRGraph(...)
    cfg = jaxsnn.event.ConversionConfig(...)

    topology = jaxsnn.event.from_nir(nir_graph, cfg)
    init, apply = topology.done()
    ```
    """
    # define topology
    topology = Topology(t_max=config.t_max)

    # create modules
    for (node_key, node) in graph.nodes.items():
        if not isinstance(node, nir.Output):  # skip output nodes
            topology.add({node_key: node_from_nir(node_key, node, config)})

    # connect modules
    # delete output nodes from edges
    edges = [edge for edge in graph.edges if not isinstance(
        graph.nodes[edge[1]], nir.Output)]
    topology.connect(edges)

    return topology
