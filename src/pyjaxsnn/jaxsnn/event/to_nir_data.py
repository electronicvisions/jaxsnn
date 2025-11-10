"""
Translate spikes from a jaxsnn spike representation to NIRGraphData.
"""

from typing import Dict

import numpy as np

from jaxsnn.event.types import EventPropSpike
from nir.data_ir import EventData, NIRGraphData, NIRNodeData, ValuedEventData


def to_nir_data(jaxsnn_dict: Dict[str, EventPropSpike], jaxsnn_model,
                observables=('spikes',)) -> NIRGraphData:
    """
    Convert a dict of EventPropSpikes (jax-snn representation) to NIRGraphData.

    :param jaxsnn_dict:
        Dictionary of Spike objects where each entry represents the spikes for
        a corresponding node of the jaxsnn_model. Empty events in jaxsnn
        are encoded by `idx = -1` and `time = 2 * t_max`.
    :param jaxsnn_model:
        A tuple of (init_fn, apply_fn). For the apply function, the arguments
        apply.nodes (which holds the layer sizes) and apply.t_max (the
        simulation time) currently have to be defined manually.
    :param observables:
        Observables to be converted, by default ('spikes',)
    """

    apply = jaxsnn_model[1]
    layer_sizes = apply.nodes

    nir_nodes = {}

    for key, spikes in jaxsnn_dict.items():
        if len(spikes.time.shape) != 2:
            raise TypeError(
                f"""Spikes must be of shape (batch_size, n_spikes) but got
                {spikes.time.shape}"""
            )
        time = np.where(spikes.time == np.inf,
                        2 * apply.t_max, spikes.time)
        idx = np.array(spikes.idx)
        nir_node_data = NIRNodeData({})
        if "spikes" in observables:
            nir_node_data.observables["spikes"] = EventData(
                idx, time, layer_sizes[key], apply.t_max)
        if "current" in observables:
            current = np.array(spikes.current)
            nir_node_data.observables["current"] = ValuedEventData(
                idx, time, layer_sizes[key], apply.t_max, current)
        nir_nodes[key] = nir_node_data

    nir_data = NIRGraphData(nir_nodes)

    return nir_data
