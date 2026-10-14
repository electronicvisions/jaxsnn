"""Conversion of a NIR graph to jaxsnn model"""

from dataclasses import dataclass
from typing import Dict, Union

import numpy as np
import jax.numpy as jnp

from jaxsnn.base.types import Parameter
from jaxsnn.event.hardware.parameter import HXParameter
from jaxsnn.event.modules import Source, LIF, Linear
from jaxsnn.event.modules.hx import HXLIF, NeuronParameters
from jaxsnn.event.modules.lif import LIFParameters
from jaxsnn.event.topology import Topology
from jaxsnn.event.types import (
    Projection,
    Population,
    SourcePopulation
)
import nir


@dataclass
class ConversionConfig:
    """
    Configuration for the conversion from NIR to jaxsnn.

    :param n_steps: Dictionary with number of spikes to simulate per layer.
    :param t_max: Maximum time for the simulation.
    :param backprop_method: Either "eventprop" for the EventProp algorithm or
        "analytical" for the time-to-first-spike solver.
    :param compute_rate: Whether to compute firing rates for the neurons.
    :param external: Whether an externally run forward pass will be injected.
    :param mock: Whether the forward pass is simulated.
    """

    n_steps: Dict[str, int]
    t_max: float
    backprop_method: str = "analytical"
    compute_rate: bool = False
    external: bool = False
    mock: bool = True


def node_from_nir(
    node_key: str,
    node: nir.NIRNode,
    config: ConversionConfig
) -> Union[Population, Projection, SourcePopulation]:
    """
    Convert a NIR node to a jaxsnn module.

    :param node_key: Key of the node in the NIR graph.
    :param node: NIR node to convert.
    :param config: Conversion configuration.

    :return: Converted jaxsnn module.
    """
    if isinstance(node, nir.Input):
        source = Source(node.input_type["input"][0])
        source.n_steps = config.n_steps[node_key]
        return source

    if isinstance(node, nir.CubaLIF):
        if node_key not in config.n_steps:
            raise KeyError(
                f"Number of event steps in the simulation for {node_key} not "
                "defined in config.n_steps"
            )

        size = np.size(node.tau_mem)
        if not config.external:
            params = LIFParameters(
                v_leak=node.v_leak[0],
                v_reset=node.v_reset[0],
                v_th=node.v_threshold[0],
                tau_mem=node.tau_mem[0] * 1e-3,
                tau_syn=node.tau_syn[0] * 1e-3,
            )
            return LIF(
                size=size,
                n_steps=config.n_steps[node_key],
                params=params,
                compute_rate=config.compute_rate
            )

        params = NeuronParameters(
            v_leak=HXParameter(node.v_leak[0]),
            v_reset=HXParameter(node.v_reset[0]),
            v_th=HXParameter(node.v_threshold[0]),
            tau_mem=HXParameter(node.tau_mem[0] * 1e-3),
            tau_syn=HXParameter(node.tau_syn[0] * 1e-3),
        )
        return HXLIF(
            size=size,
            n_steps=config.n_steps[node_key],
            params=params,
            compute_rate=config.compute_rate
        )

    if isinstance(node, nir.Linear):
        return Linear(pre_weights=jnp.asarray(node.weight.T))

    raise NotImplementedError(f"Node type {type(node)} not supported yet")


def from_nir(
    graph: nir.NIRGraph,
    config: ConversionConfig
) -> Topology:
    """
    Convert NIRGraph to jaxsnn Topology

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
    topology = Topology(
        t_max=config.t_max,
        backprop_method=config.backprop_method,
        mock=config.mock,
        has_external_events=config.external,
    )

    # create modules
    for node_key, node in graph.nodes.items():
        if not isinstance(node, nir.Output):  # skip output nodes
            topology.add({node_key: node_from_nir(node_key, node, config)})

    # connect modules
    # delete output nodes from edges
    edges = [
        edge
        for edge in graph.edges
        if not isinstance(graph.nodes[edge[1]], nir.Output)
    ]
    topology.connect(edges)

    return topology
