"""Conversion of a jaxsnn model to a NIR graph"""
from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import jax

from jaxsnn.event.types import (
    EventBaseModule,
    SourcePopulation,
    Population,
    Projection,
)
from jaxsnn.event.topology import Topology
import nir
if TYPE_CHECKING:
    from jaxsnn.event.hardware.parameter import HXBaseParameter


def node_to_nir(
    module: EventBaseModule,
    params: dict
) -> nir.NIRNode:
    """
    Convert a jaxsnn module to a NIR node.

    :param module: jaxsnn module to convert.
    :param params: Parameters for the module.

    :return: Converted NIR node.
    """
    if isinstance(module, SourcePopulation):
        return nir.Input(input_type={"input": [module.parameters["size"]]})

    # only event-based CubaLIF population for now
    if isinstance(module, Population):

        def _filled(param: Union[jax.Array, HXBaseParameter]) -> np.ndarray:
            value = (
                param
                if isinstance(param, jax.Array)
                else param.model_value
            )
            return np.full(module.parameters["size"], np.asarray(value)).T

        return nir.CubaLIF(
            v_leak=_filled(module.parameters["lif_params"].v_leak),
            v_reset=_filled(module.parameters["lif_params"].v_reset),
            v_threshold=_filled(module.parameters["lif_params"].v_th),
            tau_mem=_filled(module.parameters["lif_params"].tau_mem) * 1e3,
            tau_syn=_filled(module.parameters["lif_params"].tau_syn) * 1e3,
            r=np.ones(module.parameters["size"]),
        )

    if isinstance(module, Projection):
        return nir.Linear(weight=np.asarray(params.T))

    raise ValueError(
        f"Unsupported module type {type(module)} for NIR conversion."
    )


def to_nir(
    topology: Topology,
    params: dict,
    output: Optional[List[str]] = None
) -> nir.NIRGraph:
    """
    Convert a jaxsnn model to a NIRGraph.
    Note that the output node is not explicitly represented in the jaxsnn
    topology. Therefore, if an output node is desired in the NIRGraph, it can
    be specified explicitly via the `output` argument after which node of the
    jaxsnn topology the output node should be added.

    :param topology: jaxsnn topology to convert.
    :param params: Parameters for the model.
    :param output: Keys of the output nodes in the jaxsnn topology.

    :return: Converted NIRGraph.
    """

    nir_nodes = {}
    for node_key, node in topology.graph.nodes.items():
        nir_node = node_to_nir(node["module"], params[node_key])
        nir_nodes[node_key] = nir_node

    if output is not None:
        for node_key in output:
            size = topology.graph.nodes[node_key]["module"].parameters["size"]
            nir_nodes[f"output_{node_key}"] = nir.Output(
                output_type={"output": [size]}
            )
            nir_edges = (
                list(topology.graph.edges) + [(node_key, f"output_{node_key}")]
            )
    else:
        nir_edges = list(topology.graph.edges)

    return nir.NIRGraph(
        nodes=nir_nodes,
        edges=nir_edges,
    )
