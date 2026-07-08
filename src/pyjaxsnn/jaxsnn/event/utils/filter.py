import jax
import jax.numpy as jnp
from jaxsnn.event.types import EventPropSpike

try:
    from jax.tree import map as tree_map
except ImportError:
    # for compatibility with jax@:0.4.25
    from jax.tree_util import tree_map


def filter_spikes(
    input_spikes: EventPropSpike,
    prev_layer_start: int
) -> EventPropSpike:
    """Filters the input spikes by ensuring only the spikes from the previous
    layer are kept.
    """
    # Filter out input spikes that are not from the previous layer
    idx = input_spikes.idx >= prev_layer_start
    filtered_spikes = EventPropSpike(
        time=jnp.where(idx, input_spikes.time, jnp.inf),
        idx=jnp.where(idx, input_spikes.idx, -1),
        current=jnp.zeros_like(input_spikes.current))
    idx = jnp.argsort(filtered_spikes.time)
    input_spikes = tree_map(lambda x: x[idx], filtered_spikes)
    return input_spikes
