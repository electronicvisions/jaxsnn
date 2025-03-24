from typing import Callable, Tuple, Any, Optional

import jax
import jax.numpy as np
from jaxsnn.event.types import (
    EventPropSpike,
    InputQueue,
    LIFState,
    SingleApply,
    Solver,
    StepState,
    Weight,
)


# Input to step function.
# Consists of StepState, weights of the network and start index of the layer
StepInput = Tuple[StepState, Weight, int]


def step(  # pylint: disable=unused-argument,too-many-locals
    dynamics: Callable,
    tr_dynamics: Callable,
    t_max: float,
    solver: Solver,
    step_input: StepInput,
    *args: int,
) -> Tuple[StepInput, EventPropSpike]:
    """Find next spike (external or internal), and simulate to that point.

    Args:
        dynamics (Callable): Function describing the continuous neuron dynamics
        tr_dynamics (Callable): Function describing the transition dynamics
        t_max (float): Max time until which to run
        solver (Solver): Parallel root solver which returns the next event
        state (StepInput): (StepState, weights, int)
    Returns:
        Tuple[StepInput, Spike]: New state after transition and stored spike
    """
    state, weights, layer_start = step_input
    prev_layer_start = layer_start - weights.input.shape[0]

    def return_existing_event(current_state):
        """
        We have queued spikes because multiple spikes occurred at the same time

        TODO: Think about spikes later than t_max
        """
        idx = np.argmin(current_state.spike_times)
        time = current_state.spike_times[idx]

        no_event = time >= t_max
        stored_idx = jax.lax.cond(
            no_event, lambda: -1, lambda: idx + layer_start)

        # Event to return
        new_event = EventPropSpike(
            time, stored_idx, current_state.neuron_state.I[idx])

        # Update masks
        current_state.spike_times \
            = current_state.spike_times.at[idx].set(t_max)
        current_state.spike_mask \
            = current_state.spike_mask.at[idx].set(False)

        return (current_state, weights, layer_start), new_event

    def find_new_events(current_state):
        next_times = solver(
            current_state.neuron_state, current_state.time, t_max)
        next_internal_idx = np.argmin(next_times)
        next_internal_time = np.minimum(next_times[next_internal_idx], t_max)

        # determine spike nature and spike time
        input_time = jax.lax.cond(
            current_state.input_queue.is_empty,
            lambda: t_max,
            lambda: current_state.input_queue.peek().time)
        t_dyn = np.minimum(next_internal_time, input_time)

        # comparing only makes sense if exactly dt is returned from solver
        spike_in_layer = next_internal_time < input_time
        no_event = t_dyn >= t_max

        # New neuron state
        evolved_neuron_state = dynamics(
            current_state.neuron_state, t_dyn - current_state.time)

        # Detect where neurons have spiked
        spike_mask = np.zeros_like(
            current_state.spike_mask).at[next_internal_idx].set(True)
        spike_mask = np.where(
            # Sometimes other neurons cross threshold in evolved state because
            # of numerical differences
            (evolved_neuron_state.V >= 1.)
            # Some times neurons are very close to threshold but smaller when
            # multiple neurons spike simultaneously
            | (next_times == next_internal_time),
            True, spike_mask)
        spike_mask = np.where(
            no_event | ~spike_in_layer,
            np.zeros_like(current_state.spike_times, dtype=bool),
            spike_mask)

        # Create new event
        current = jax.lax.cond(
            spike_in_layer,
            lambda: evolved_neuron_state.I[next_internal_idx],
            lambda: current_state.input_queue.peek().current)

        stored_idx = jax.lax.cond(
            no_event,
            lambda: -1,
            lambda: jax.lax.cond(
                spike_in_layer,
                lambda: next_internal_idx + layer_start,
                lambda: current_state.input_queue.peek().idx))

        new_event = EventPropSpike(t_dyn, stored_idx, current)

        # Update step state
        evolved_state = StepState(
            neuron_state=evolved_neuron_state,
            spike_times=next_times,
            spike_mask=spike_mask,
            time=t_dyn,
            input_queue=current_state.input_queue)

        # Transition state
        transitioned_state = jax.lax.cond(
            no_event,
            lambda *args: evolved_state,
            tr_dynamics,
            evolved_state,
            weights,
            spike_mask,
            spike_in_layer,
            prev_layer_start)

        transitioned_state.spike_times \
            = next_times.at[next_internal_idx].set(t_max)
        transitioned_state.spike_mask \
            = spike_mask.at[next_internal_idx].set(False)

        return (transitioned_state, weights, layer_start), new_event

    return jax.lax.cond(
        np.any(state.spike_mask),
        return_existing_event,
        find_new_events,
        state)


def trajectory(
    step_fn: Callable[[StepInput, int], Tuple[StepInput, EventPropSpike]],
    n_hidden: int,
    n_spikes: int,
) -> SingleApply:
    """Evaluate the `step_fn` until `n_spikes` have been simulated.

    Uses a scan over the `step_fn` to return an apply function
    """

    def apply_fn(  # pylint: disable=unused-argument
        weights: Weight,
        input_spikes: EventPropSpike,
        external: Any,
        carry: Optional[int],
    ) -> Tuple[int, Weight, EventPropSpike, EventPropSpike]:
        if carry is None:
            layer_index = 0
            layer_start = 0
        else:
            layer_index, layer_start = carry
        this_layer_weights = weights[layer_index]
        input_size = this_layer_weights.input.shape[0]
        layer_start = layer_start + input_size

        # Filter out input spikes which are not from previous layer
        input_spikes = filter_spikes(input_spikes, layer_start - input_size)

        step_state = StepState(
            neuron_state=LIFState(
                np.zeros(n_hidden), np.zeros(n_hidden)),
            spike_times=-1 * np.ones(n_hidden),
            spike_mask=np.zeros(n_hidden, dtype=bool),
            time=0.0,
            input_queue=InputQueue(input_spikes),
        )
        _, spikes = jax.lax.scan(
            step_fn,
            (step_state, this_layer_weights, layer_start),
            np.arange(n_spikes)
        )

        layer_index += 1
        return (layer_index, layer_start), this_layer_weights, spikes, spikes

    return apply_fn


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
        time=np.where(idx, input_spikes.time, np.inf),
        idx=np.where(idx, input_spikes.idx, -1),
        current=np.zeros_like(input_spikes.current))
    idx = np.argsort(filtered_spikes.time)
    input_spikes = jax.tree_map(lambda x: x[idx], filtered_spikes)
    return input_spikes
