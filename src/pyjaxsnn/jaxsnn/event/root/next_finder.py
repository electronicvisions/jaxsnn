from typing import Callable

import jax.numpy as np
from jaxsnn.event.types import LIFState, Spike


def next_event(
    solver: Callable, neuron_state: LIFState, time: float, t_max: float
) -> Spike:
    """Wrapper a root solver to provide a cleaner API for returning next event

    Args:
        solver (Callable): The actual root solver
        neuron_state (LIFState): The state of the neurons
        time (float): Current time
        t_max (float): Maximum time of the simulation

    Returns:
        Spike: Spike which will occur next
    """
    pred_spikes = solver(neuron_state, t_max) + time
    idx = np.argmin(pred_spikes)
    return Spike(pred_spikes[idx], idx=idx)


def next_queue(
    known_spikes: Spike,
    layer_start: int,
    neuron_state: LIFState,  # pylint: disable=unused-argument
    time: float,
    t_max: float,
) -> Spike:
    """Return the upcoming spike when training with hardware-in-the-loop.

    When working with the BSS-2 system, we have all the spikes in advance
    and need to find the index and time of the next event. When the hardware
    spikes are bound to this function with `functools.partial`, it has the
    same API as `next_event`.

    Args:
        known_spikes (Spike): All spikes from BSS-2
        layer_start (int): Start index of the current layer
        neuron_state (LIFState): The state of the neurons
        time (float): Current time
        t_max (float): max time

    Returns:
        Spike: Spike which will occur next in the layer
    """
    this_layer = np.where(
        known_spikes.idx >= layer_start, known_spikes.time, t_max
    )
    time_or_t_max = np.where(this_layer > time, this_layer, t_max)
    idx = np.argmin(time_or_t_max)
    return Spike(time_or_t_max[idx], known_spikes.idx[idx] - layer_start)
