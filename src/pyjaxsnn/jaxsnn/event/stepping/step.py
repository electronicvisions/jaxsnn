from typing import (
    List,
    Callable,
    Tuple,
    Dict,
)
from dataclasses import replace

import jax
import jax.numpy as jnp

from jaxsnn.event.types import (
    EventT,
    DynamicsFn,
    MinDelayCheckFn,
    NextInputFn,
    Step,
    StepState,
    Spike,
    SolverFn,
    QueueHead,
    QueueIndex,
)


def next_input(  # pylint: disable=too-many-locals,too-many-arguments
    input_layers: List[str],
    min_delays: Dict[str, float],
    spikes: Dict[str, Spike],
    queue_heads: jax.Array,
    time: float,
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
            & (layer_spikes.time + min_delays[input_node] >= time)
            & (jnp.arange(layer_spikes.time.size) >= queue_heads[i])
        )

        allowed_times = jnp.where(mask, layer_spikes.time, t_max)
        idx = jnp.argmin(allowed_times)

        # Store the per-layer results
        next_times = next_times.at[i].set(
            allowed_times[idx] + min_delays[input_node],
        )
        queue_idx = queue_idx.at[i].set(idx)
        spike_candidates = spike_candidates.set_item(i, layer_spikes[idx])

    earliest_layer = jnp.argmin(next_times)
    spike_idx = queue_idx[earliest_layer]

    final_spike = replace(
        spike_candidates[earliest_layer],
        time=next_times[earliest_layer],
    )

    return earliest_layer, spike_idx, final_spike


def min_delay_check(
    input_nodes: List[str],
    min_delays: Dict[str, float],
    spikes: Dict[str, EventT],
    spike_time: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """
    Check if a given spike time is not too far ahead in the future.

    For each recurrently connected input layer, finds the latest spike
    time plus the minimum delay, then compares to the given spike time.
    This ensures that no input spike can reach the neuron before the
    time of the spike.

    :param input_nodes: List of indices of input layers.
    :param min_delays: Array of minimum delays per input connection.
    :param spikes: List of spike records for all layers.
    :param spike_time: The spike time to check.

    :returns: Tuple containing
        - boolean indicating if spike_time is safe,
        - the computed safe time (minimum allowed spike time).
    """
    if len(input_nodes) == 0:
        return jnp.array(True), spike_time

    next_times = jnp.zeros(len(input_nodes))
    for i, input_node in enumerate(input_nodes):
        layer_spikes = spikes[input_node]

        mask = layer_spikes.idx != -1
        allowed_times = jnp.where(mask, layer_spikes.time, 0)
        idx = jnp.argmax(allowed_times)

        next_times = next_times.at[i].set(
            allowed_times[idx] + min_delays[input_node],
        )

    safe_time = jnp.min(next_times)

    return safe_time >= spike_time, safe_time


def step(  # pylint: disable=unused-argument,too-many-locals,too-many-arguments
    next_input_fn: NextInputFn,
    min_delay_check_fn: MinDelayCheckFn,
    dynamics: DynamicsFn,
    tr_dynamics: List[Callable],
    t_max: float,
    solver: SolverFn,
    step_input: Step,
) -> Tuple[Spike, StepState, QueueHead, QueueIndex]:
    """
    Find the next spike (external or internal), and evolve state to that point.

    :param next_input_fn: Function to find the next input spike.
    :param min_delay_check_fn: Function to check if spike is allowed.
    :param dynamics: Function describing continuous neuron dynamics.
    :param tr_dynamics: List of functions describing the transition after a
        spike.
    :param t_max: Maximum simulation time for this step.
    :param solver: Solver that returns the next internal event.
    :param step_input: Tuple containing
        (weights, spikes, state, _, layer_idx, queue_heads, _).

    :returns: Tuple containing
        - Spike: The new spike event.
        - StepState: Updated neuron state after transition.
        - QueueHead: Updated queue heads.
        - QueueIndex: Spike queue index of the spike event.
    """
    next_internal = solver(
        step_input.state.neuron_state,
        step_input.state.time,
        t_max,
    )

    # Find next input spike
    rel_input_layer_idx, spike_queue_idx, input_spike = next_input_fn(
        step_input.spikes,
        step_input.queue_head,
        step_input.state.time,
        t_max,
    )

    t_dyn = jnp.minimum(next_internal.time, input_spike.time)

    spike_allowed, safe_time = min_delay_check_fn(step_input.spikes, t_dyn)
    t_dyn = jax.lax.cond(
        spike_allowed, lambda: t_dyn, lambda: safe_time,
    )

    spike_in_layer = next_internal.time < input_spike.time

    no_event = jnp.logical_or(
        t_dyn >= t_max, jnp.logical_not(spike_allowed))

    internal = jnp.logical_and(jnp.logical_not(no_event), spike_in_layer)

    queue_head = jax.lax.cond(
        jnp.logical_or(spike_in_layer, no_event),
        lambda: step_input.queue_head,
        lambda: step_input.queue_head.at[rel_input_layer_idx].set(
            spike_queue_idx + 1,
        ),
    )

    stored_layer_idx = jax.lax.cond(
        no_event,
        lambda: -1,
        lambda: jax.lax.cond(
            spike_in_layer,
            lambda: step_input.layer_idx,
            lambda: input_spike.layer_idx,
        ),
    )
    stored_internal_idx = jax.lax.cond(
        no_event,
        lambda: -1,
        lambda: jax.lax.cond(
            spike_in_layer,
            lambda: next_internal.idx,
            lambda: input_spike.idx,
        ),
    )

    state: StepState = jax.lax.cond(
        no_event,
        lambda: step_input.state,
        lambda: StepState(
            neuron_state=dynamics(
                step_input.state.neuron_state,
                t_dyn - step_input.state.time,
            ),
            time=t_dyn,
        ),
    )

    current = jax.lax.cond(
        spike_in_layer,
        lambda: state.neuron_state.I[next_internal.idx],
        lambda: input_spike.current,
    )

    transitioned_state = jax.lax.cond(
        no_event,
        lambda: state,
        lambda: jax.lax.switch(
            rel_input_layer_idx,
            tr_dynamics,
            state,
            step_input.parameters,
            stored_internal_idx,
            spike_in_layer,
        ),
    )

    stored_time = jax.lax.cond(
        no_event,
        lambda: t_max,
        lambda: t_dyn,
    )

    return (
        Spike(
            time=stored_time,
            idx=stored_internal_idx,
            current=current,
            layer_idx=stored_layer_idx,
            internal=internal,
        ),
        transitioned_state,
        queue_head,
        spike_queue_idx,
    )
