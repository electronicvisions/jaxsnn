from typing import Callable, List, Tuple, Dict, Optional

import jax
import jax.numpy as jnp
from jaxsnn.event.types import (
    Observable,
    Observables,
    TrajectoryCarry,
    IOData,
    Parameters,
    States,
)


def trajectory(  # pylint: disable=too-many-arguments
    multi_layer_step_fn: Callable,
    n_steps: int,
    compute_rates: Dict[str, int],
    parameters: Parameters,
    observables: Observables,
    external_spikes: Optional[IOData],
    states: States,
    queue_heads: Dict[str, jax.Array],
) -> Tuple[IOData, States, Parameters, List[jax.Array]]:
    """
    Simulate over multiple time steps for recurrent sub-networks.

    Applies the multi-layer step function sequentially across `n_steps`
    using JAX's scan. Maintains the states, spike history, and queue indices.

    :param multi_layer_step_fn: Function to advance all layers one time step.
    :param n_steps: Number of simulation steps.
    :param parameters: Model parameters.
    :param spikes: Spikes from all layers.
    :param states: Initial neuron states.
    :param queue_heads: Dict of queue head arrays for input queuing.

    :returns: Tuple of updated (spikes, states, parameters, queue_indices).
    """
    queue_indices = {
        node: jnp.zeros(n_steps, dtype=int) for node in queue_heads
    }
    carry = TrajectoryCarry(
        parameters,
        observables,
        external_spikes,
        states,
        queue_heads,
        queue_indices,
    )
    carry, _ = jax.lax.scan(
        multi_layer_step_fn,
        carry,
        jnp.arange(n_steps),
    )

    # Compute firing rates if required
    for node, value in carry.observables.items():
        if value is None:
            continue
        rate = value.rate
        if node in compute_rates:
            rate, _ = jnp.histogram(
                jnp.where(value.spike.internal, value.spike.idx, -1),
                bins=jnp.arange(compute_rates[node] + 1),
            )
        carry.observables[node] = Observable(value.spike, rate)

    return (
        carry.observables,
        carry.states,
        carry.parameters,
        carry.queue_indices,
    )
