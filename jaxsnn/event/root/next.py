from typing import Callable
from jaxsnn.neuron import NeuronState
from jaxsnn.base.types import Spike
import jax.numpy as np


def next_event(
    solver: Callable, neuron_state: NeuronState, time: float, t_max: float
) -> Spike:
    """Wrap a root solver to return the next event"""
    pred_spikes = solver(neuron_state, t_max) + time
    idx = np.argmin(pred_spikes)
    return Spike(pred_spikes[idx], idx=idx)


def next_queue(
    known_spikes: Spike,
    layer_start: int,
    neuron_state: NeuronState,
    time: float,
    t_max: float,
) -> Spike:
    """Return the spike which is the next one coming up

    When working with hardware, we have all the spikes in advance
    and need to finde the index and time of the next event.

    If no event is ahead of time anymore, return t_max.

    Args:
        known_spikes (Spike): All spikes in layer known in advance
        time (float): current time
        t_max (float): max time

    Returns:
        Spike: Next Spikes
    """
    this_layer = np.where(known_spikes.idx >= layer_start, known_spikes.time, t_max)
    time_or_t_max = np.where(this_layer > time, this_layer, t_max)
    idx = np.argmin(time_or_t_max)
    return Spike(time_or_t_max[idx], known_spikes.idx[idx] - layer_start)
