"""
Translate NIRGraphData to a jaxsnn-compatible spike representation.
"""

from typing import Dict

import jax.numpy as jnp
import numpy as np

from jaxsnn.event.types import EventPropSpike
from nir.data_ir import (
    TimeGriddedData,
    NIRGraphData,
    NIRNodeData,
)


def from_nir_data(nir_graph_data: NIRGraphData, jaxsnn_model,
                  ) -> Dict[str, EventPropSpike]:
    """
    Convert NIRGraphData to a dict of EventPropSpikes (jax-snn representation)

    A linear noise is added on the spike times if the incoming data is
    time-gridded.
    """

    _, apply = jaxsnn_model
    jaxsnn_dict = {}

    for node_key, nir_node_data in nir_graph_data.nodes.items():
        if isinstance(nir_node_data, NIRNodeData):
            for observable, data in nir_node_data.observables.items():
                if observable == 'spikes':
                    if isinstance(data, TimeGriddedData):
                        data = data.to_event(n_spikes=apply.nodes[node_key])

                    data.time = jnp.asarray(data.time)
                    jaxsnn_time = jnp.where(data.time == np.inf,
                                            2 * data.t_max,
                                            data.time)
                    jaxsnn_spikes = EventPropSpike(
                        jaxsnn_time,
                        jnp.asarray(data.idx),
                        jnp.zeros_like(jnp.asarray(data.idx),
                                       dtype=jaxsnn_time.dtype)  # pylint: disable=no-member
                    )

                    jaxsnn_dict[node_key] = jaxsnn_spikes
                else:
                    raise NotImplementedError('Only spikes are supported as '
                                              'observables yet.')
        else:
            raise NotImplementedError('The translation of nested NIRGraphData'
                                      'is not supported.')

    return jaxsnn_dict
