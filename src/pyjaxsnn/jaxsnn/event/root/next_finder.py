from typing import Callable

import jax
from jaxsnn.event.types import LIFState


def next_event(
    solver: Callable, neuron_state: LIFState, time: float, t_max: float
) -> jax.Array:
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
    return pred_spikes
