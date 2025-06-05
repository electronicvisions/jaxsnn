import jax
import jax.numpy as jnp

from jaxsnn.base.types import BaseState
from jaxsnn.event.types import Event, SolverFn


def next_event(
    solver: SolverFn,
    neuron_state: BaseState,
    time: jax.Array,
    t_max: float,
) -> Event:
    """
    Wraps a root solver to provide a cleaner API for returning next event.

    :param solver: The actual root solver function.
    :param neuron_state: The state of the neurons.
    :param time: Current simulation time.
    :param t_max: Maximum time of the simulation.

    :returns: Event object representing the spike which will occur next.
    """
    pred_spikes = solver(neuron_state, t_max) + time
    idx = jnp.argmin(pred_spikes)
    return Event(pred_spikes[idx], idx)


def next_queue(
    known_spikes: Event,
    layer_start: int,
    neuron_state: BaseState,  # pylint: disable=unused-argument
    time: float,
    t_max: float,
) -> Event:
    """
    Return the upcoming spike when training with hardware-in-the-loop.

    When working with the BSS-2 system, we have all the spikes in advance
    and need to find the index and time of the next event. When the hardware
    spikes are bound to this function with `functools.partial`, it has the
    same API as `next_event`.

    :param known_spikes: All spikes from BSS-2.
    :param layer_start: Start index of the current layer.
    :param neuron_state: The state of the neurons (unused).
    :param time: Current simulation time.
    :param t_max: Maximum simulation time.

    :returns: Event object representing the spike which will occur next in the
        layer.
    """
    this_layer = jnp.where(
        known_spikes.idx >= layer_start, known_spikes.time, t_max
    )
    time_or_t_max = jnp.where(this_layer > time, this_layer, t_max)
    idx = jnp.argmin(time_or_t_max)
    return Event(time_or_t_max[idx], known_spikes.idx[idx] - layer_start)
