"""
Translate spikes from a jaxsnn spike representation to NIRGraphData.
"""

from typing import Dict

import numpy as np

from jaxsnn.event.types import EventPropSpike
from nir.data_ir import EventData, NIRGraphData, NIRNodeData


def to_nir_data(jaxsnn_dict: Dict[str, EventPropSpike], jaxsnn_model
                ) -> NIRGraphData:
    """
    :param jaxsnn_dict: Dictionary of Spike objects where each entry represents
        the spikes for a corresponding input node of the jaxsnn_model. Empty
        events in jaxsnn are encoded by idx = -1 and time = np.inf.
    :param jaxsnn_model: A tuple of (init_fn, apply_fn).
    """

    apply = jaxsnn_model[1]
    jaxsnn_nodes = apply.nodes

    nir_nodes = {}

    for key, spikes in jaxsnn_dict.items():
        if len(spikes.time.shape) != 2:
            raise TypeError(
                f'''Spikes must be of shape (batch_size, n_spikes) but got
                {spikes.time.shape}'''
            )
        time = np.where(spikes.time == np.inf,
                        2 * apply.t_max, spikes.time)
        idx = np.array(spikes.idx)
        nir_node_data = NIRNodeData({'spikes': EventData(idx, time,
                                                         jaxsnn_nodes[key],
                                                         apply.t_max)})
        nir_nodes[key] = nir_node_data

    nir_data = NIRGraphData(nir_nodes)

    return nir_data
