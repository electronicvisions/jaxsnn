from typing import (
    Callable,
    Tuple,
    List,
    Dict,
)
from dataclasses import replace

import jax
import jax.numpy as jnp

from jaxsnn.event.types import (
    Spike,
    QueueHead,
    Step,
    StepState,
    QueueIndex,
)


def next_input(  # pylint: disable=too-many-locals,too-many-arguments
    input_layers: List[str],
    spikes: Dict[str, Spike],
    queue_heads: jax.Array,
    t: float,  # pylint: disable=invalid-name
    t_max: float,
) -> Tuple[jax.Array, jax.Array, Spike]:
    """
    Find the next input spike event across potentially multiple input layers.
    """
    next_times = jnp.zeros(len(input_layers))
    queue_idx = jnp.zeros(len(input_layers), dtype=int)

    # Create a template PyTree for each layer
    spike_candidates = spikes[input_layers[0]].empty(
        shape=(len(input_layers),)
    )
    for i, input_node in enumerate(input_layers):
        layer_spikes = spikes[input_node]

        mask = (
            layer_spikes.internal
            & (layer_spikes.time >= t)
            & (jnp.arange(layer_spikes.time.size) >= queue_heads[i])
        )

        allowed_times = jnp.where(mask, layer_spikes.time, t_max)
        idx = jnp.argmin(allowed_times)

        # Store the per-layer results
        next_times = next_times.at[i].set(allowed_times[idx])
        queue_idx = queue_idx.at[i].set(idx)
        spike_candidates = spike_candidates.set_item(i, layer_spikes[idx])

    earliest_layer = jnp.argmin(next_times)
    spike_idx = queue_idx[earliest_layer]

    final_spike = replace(
        spike_candidates[earliest_layer],
        time=next_times[earliest_layer],
    )

    return earliest_layer, spike_idx, final_spike


def step_existing(  # pylint: disable=too-many-locals,too-many-arguments
        input_layers: List[str],
        dynamics: Callable,
        transition_fns: List[Callable],
        node: str,
        t_max: float,
        step_input: Step,
) -> Tuple[Spike, StepState, QueueHead, QueueIndex]:
    """
    Simulate a single event-driven step for a spiking neuron layer using
    existing (from hardware) events.

    Finds the next relevant spike (known hardware spike or input event),
    advances the neuron state to that event, applies the appropriate
    transition, and updates all event queues and indices accordingly.

    :param input_layers: List of input layer names.
    :param valid_input_layer_indices: Indices of valid input layers for event
        matching.
    :param dynamics: Function describing the continuous neuron dynamics.
    :param transition_fns: List of transition functions for each possible
        input.
    :param node: Name of the current layer/node.
    :param t_max: Maximum simulation time for this step.
    :param step_input: Step object containing all state, parameters, queues,
        and indices.

    :returns: Tuple containing:
        - Spike: The spike event at this step (or empty if none).
        - StepState: Updated neuron state after the event.
        - QueueHead: Updated queue head indices for all inputs.
        - QueueIndex: Index of the input queue used for the event.
    """
    assert step_input.external_spikes is not None

    queue_head = step_input.queue_head

    # next event
    internal_spikes = step_input.external_spikes[node]
    internal_queue_idx = queue_head[-1]
    next_internal = internal_spikes[internal_queue_idx]

    # Find next input spike
    rel_input_layer_idx, input_queue_idx, input_spike = next_input(
        input_layers,
        step_input.spikes,
        queue_head,
        step_input.state.time,
        t_max,
    )

    # Set flags
    next_internal_time = jnp.where(
        internal_queue_idx >= internal_spikes.time.shape[0],
        t_max,
        next_internal.time,
    )

    event_time = jnp.minimum(next_internal_time, input_spike.time)
    is_internal_event = next_internal_time < input_spike.time
    no_event = event_time >= t_max
    event_time = jnp.where(no_event, t_max, event_time)

    # Increment queue heads
    updated_input_queue_head = \
        queue_head.at[rel_input_layer_idx].set(input_queue_idx + 1)
    updated_internal_queue_head = \
        queue_head.at[-1].set(internal_queue_idx + 1)
    queue_head = jnp.where(
        is_internal_event,
        updated_internal_queue_head,
        updated_input_queue_head,
    )

    # Layer idx
    layer_idx = jnp.where(
        no_event,
        -1,
        jnp.where(
            is_internal_event,
            next_internal.layer_idx,
            input_spike.layer_idx,
        ),
    )

    # Internal spike idx
    idx = jnp.where(
        no_event,
        -1,
        jnp.where(
            is_internal_event,
            next_internal.idx,
            input_spike.idx,
        ),
    )

    # Update interal
    internal = jnp.where(
        no_event,
        False,
        jnp.where(
            is_internal_event,
            True,
            False,
        ),
    )

    # Updates neuron state
    neuron_state = jax.lax.cond(
        no_event,
        lambda: step_input.state.neuron_state,
        lambda: dynamics(
            step_input.state.neuron_state,
            event_time - step_input.state.time,
        ),
    )
    updated_step_state = StepState(
        neuron_state=neuron_state,
        time=event_time,
    )

    # Update current
    current = jnp.where(
        no_event,
        0.0,
        jnp.where(
            is_internal_event,
            neuron_state.I[idx],
            0.0,  # input_spike.current
        ),
    )

    transitioned_neuron_state = jax.lax.cond(
        no_event,
        lambda: neuron_state,
        lambda: jax.lax.switch(
            layer_idx,
            transition_fns,
            updated_step_state,
            step_input.parameters,
            idx,
            is_internal_event,
        ).neuron_state,
    )

    return (
        Spike(
            time=event_time,
            idx=idx,
            current=current,
            layer_idx=layer_idx,
            internal=internal,
        ),
        StepState(
            neuron_state=transitioned_neuron_state,
            time=event_time,
        ),
        queue_head,
        input_queue_idx,
    )
