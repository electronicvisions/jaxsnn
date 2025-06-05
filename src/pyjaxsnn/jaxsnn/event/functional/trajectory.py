from typing import Callable, List, Tuple, Dict, Optional

import jax
import jax.numpy as jnp
from jaxsnn.event.types import (
    Carry,
    IOData,
    Parameters,
    States,
)


def trajectory(  # pylint: disable=too-many-arguments
    multi_layer_step_fn: Callable,
    n_steps: int,
    parameters: Parameters,
    spikes: IOData,
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
    carry = Carry(
        parameters,
        spikes,
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

    return carry.spikes, carry.states, carry.parameters, carry.queue_indices
