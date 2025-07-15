"""Implement conversion of a NIR graph to jaxsnn-model"""
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import nir
from jaxsnn.base.compose import serial
from jaxsnn.event.modules.leaky_integrate_and_fire import (
    LIF, HardwareLIF, LIFParameters)
from jaxsnn.event.types import WeightInput


NUMBER_TYPES = (int, float, np.float32)

DUMMY_LAYER = (nir.Input, nir.Output)
NEURON_LAYER = (nir.CubaLIF, nir.IF, nir.LIF, nir.LI)
SYN_LAYER = (nir.Affine, nir.Linear)


@dataclass
class ConversionConfig:
    """
    Configuration for the conversion from NIR to jaxsnn.

    Attributes:
        t_max: Maximum time for the simulation.
        n_spikes: Dictionary with number of spikes to simulate per layer.
        external: If True, forward pass is executed externally.
        duplication: Number of duplications for the external hardware LIF.
    """
    t_max: float
    n_spikes: Dict[str, int]
    external: bool = False
    duplication: Optional[int] = None


def bias_is_zero(obj):
    return np.all(np.array(obj) == 0)


def is_homogeneous(arr):
    return all(x == arr[0] for x in arr)


def get_edge(graph, pre_node='', post_node=''):
    '''
    Return list of edges either starting with pre_node or ending with
    post_node
    '''
    if post_node == '':
        found_edges = [edge for edge in graph.edges if edge[0] == pre_node]
    else:
        found_edges = [
            edge for edge in graph.edges if edge[1] == post_node]
    if len(found_edges) != 1:
        raise NotImplementedError(
            "Only linear feed-forward graphs are supported yet")
    return found_edges[0]


def convert_to_number(param):
    '''
    Convert parameter to number
    '''
    if isinstance(param, np.ndarray):
        if not is_homogeneous(param):
            raise ValueError(
                "Paramaters may not differ for neurons of the same layer.")
        param = param[0]
    else:
        raise TypeError(
            f"Parameter {param} is not a numpy array.")
    return param


def get_keys(graph, node_class):
    '''
    Return array of keys of nodes of node class
    '''
    key_array = []
    for key, node in graph.nodes.items():
        if isinstance(node, node_class):
            key_array.append(key)

    return key_array


def get_prev_node(graph, node_key):
    '''
    Return previous node of node
    '''
    edge = get_edge(graph, post_node=node_key)
    prev_node = graph.nodes[edge[0]]

    return prev_node


def convert_cuba_lif(graph, node_key, config: ConversionConfig):  # pylint: disable=too-many-locals
    '''
    Convert nir.CubaLIF to jaxsnn representation (init_fn, apply_fn)
    jaxsnn representation is either a HardwareLIF or LIF layer
    '''
    node = graph.nodes[node_key]
    n_spikes = config.n_spikes[node_key]

    size = np.size(node.tau_mem)
    tau_mem = convert_to_number(node.tau_mem)
    tau_syn = convert_to_number(node.tau_syn)
    r = convert_to_number(node.r)  # pylint: disable=invalid-name
    v_leak = convert_to_number(node.v_leak)
    v_reset = convert_to_number(node.v_reset)
    v_threshold = convert_to_number(node.v_threshold)

    if r != 1:
        raise NotImplementedError('R has to be 1')

    # insert params
    params = LIFParameters(
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        v_leak=v_leak,
        v_reset=v_reset,
        v_th=v_threshold,
    )

    if config.external:
        CubaLIF = HardwareLIF  # pylint: disable=invalid-name
    else:
        CubaLIF = LIF  # pylint: disable=invalid-name

    init, apply = CubaLIF(
        size=size,
        n_spikes=n_spikes,
        t_max=config.t_max,
        params=params,
        mean=1,  # set via jaxsnn_weights_list
        std=0,
        duplication=config.duplication
    )

    return init, apply


def from_nir(graph: nir.NIRGraph, config: ConversionConfig):  # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    """
    Convert NIRGraph to jax-snn representation (init_fn, apply_fn)

    Restrictions for NIRGraph:
    - Only linear feed-forward SNNs are supported
    - CubaLIF and Linear layers are supported
    - Affine layers with bias==0 are currently supported
    - In terms of parameters, only homogeneous layers are supported

    Example:
    ```python
    nir_graph = nir.NIRGraph(...)
    cfg = jaxsnn.ConversionConfig(...)
    init, apply = jaxsnn.from_nir(nir_graph, cfg)
    ```
    """

    # assert that n_spikes is defined for all neuron layers
    for node_key in get_keys(graph, nir.CubaLIF):
        if node_key not in config.n_spikes:
            raise KeyError(
                f"n_spikes for {node_key} not defined in config.n_spikes")

    for edge in graph.edges:
        if len(edge) != 2:
            raise ValueError("All edges must have length 2")
        if not (edge[0] in graph.nodes and edge[1] in graph.nodes):
            raise KeyError(
                "The nodes used in edges must be defined in graph.nodes")

    # build list of nodes and iterate
    nodes_list = get_keys(graph, nir.Input)

    # build list of nodes
    while True:
        next_edge = get_edge(graph, pre_node=nodes_list[-1])
        nodes_list.append(next_edge[1])

        if next_edge[1] == get_keys(graph, nir.Output)[0]:
            break

    # iterate over nodes and build jaxsnn layers
    init_and_apply = []
    jaxsnn_weight_list = []

    for node_key in nodes_list:
        node = graph.nodes[node_key]
        if isinstance(node, nir.CubaLIF):
            init, apply = convert_cuba_lif(graph, node_key, config)

            # support of neuron-neuron layers by inserting dummy synapse
            # check for synapse as prev layer
            if isinstance(get_prev_node(graph, node_key), NEURON_LAYER):
                # insert dummy synapse -> unitary matrix
                unitary = np.diag(np.ones(len(node.tau_mem)))
                jaxsnn_weight = WeightInput(unitary)
                jaxsnn_weight_list.append(jaxsnn_weight)

            init_and_apply.append((init, apply))

        elif isinstance(node, nir.Linear) or (
                isinstance(node, nir.Affine) and bias_is_zero(node.bias)):
            nir_weight = node.weight
            if not isinstance(nir_weight, np.ndarray):
                nir_weight = nir_weight.numpy()
            jaxsnn_weight = WeightInput(nir_weight.T)
            jaxsnn_weight_list.append(jaxsnn_weight)

        elif isinstance(node, nir.Affine):
            if not bias_is_zero(node.bias):
                raise NotImplementedError(
                    "Affine layer with bias!=0 are currently not supported")
        elif isinstance(node, DUMMY_LAYER):
            pass
        else:
            raise NotImplementedError(
                f"{type(node)} layer currently not supported")

    _, apply_fn = serial(*init_and_apply)

    def init_fn(unused1, unused2):  # pylint: disable=unused-argument
        return unused1, jaxsnn_weight_list

    return init_fn, apply_fn
