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
                  observables=('spikes',)) -> Dict[str, EventPropSpike]:
    """
    Convert NIRGraphData to a dict of EventPropSpikes (jax-snn representation)

    :param nir_graph_data:
        NIRGraphData to be converted.
    :param jaxsnn_model:
        jaxsnn model tuple (init, apply).
    :param observables:
        Observables to be converted, by default ('spikes',)
    """

    _, apply = jaxsnn_model
    jaxsnn_dict = {}

    for node_key, nir_node_data in nir_graph_data.nodes.items():
        if isinstance(nir_node_data, NIRNodeData):
            if "spikes" in nir_node_data.observables and \
                    "spikes" in observables:
                spikes = nir_node_data.observables["spikes"]
                if isinstance(spikes, TimeGriddedData):
                    spikes = spikes.to_event(n_events=apply.nodes[node_key])

                spikes.time = jnp.asarray(spikes.time)
                jaxsnn_time = jnp.where(spikes.time == np.inf,
                                        2 * spikes.t_max,
                                        spikes.time)

                if "current" in nir_node_data.observables and \
                        "current" in observables:
                    events = nir_node_data.observables["current"]
                    if isinstance(events, TimeGriddedData):
                        raise NotImplementedError(
                            'Conversion of valued TimeGriddedData is not'
                            'supported yet.')
                    current = jnp.asarray(events.values)
                else:
                    current = jnp.zeros_like(jnp.asarray(spikes.idx),
                                             dtype=jaxsnn_time.dtype)
                jaxsnn_spikes = EventPropSpike(
                    jaxsnn_time, jnp.asarray(spikes.idx), current)

                jaxsnn_dict[node_key] = jaxsnn_spikes

        else:
            raise NotImplementedError('The translation of nested NIRGraphData'
                                      'is not supported.')

    return jaxsnn_dict
