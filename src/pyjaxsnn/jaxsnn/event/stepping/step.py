from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jaxsnn.event.types import EventPropSpike, Solver, StepState
from jaxsnn.event.stepping.types import StepInput


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
        idx = jnp.argmin(current_state.spike_times)
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
        next_internal_idx = jnp.argmin(next_times)
        next_internal_time = jnp.minimum(next_times[next_internal_idx], t_max)

        # determine spike nature and spike time
        input_time = jax.lax.cond(
            current_state.input_queue.is_empty,
            lambda: t_max,
            lambda: current_state.input_queue.peek().time)
        t_dyn = jnp.minimum(next_internal_time, input_time)

        # comparing only makes sense if exactly dt is returned from solver
        spike_in_layer = next_internal_time < input_time
        no_event = t_dyn >= t_max

        # New neuron state
        evolved_neuron_state = dynamics(
            current_state.neuron_state, t_dyn - current_state.time)

        # Detect where neurons have spiked
        # TODO: EA 2025-04-23: This is problematic for hardware because the
        # threshold test is not working reliably anymore
        spike_mask = jnp.zeros_like(
            current_state.spike_mask).at[next_internal_idx].set(True)
        spike_mask = jnp.where(
            # Sometimes other neurons cross threshold in evolved state because
            # of numerical differences
            (evolved_neuron_state.V >= 1.)
            # Some times neurons are very close to threshold but smaller when
            # multiple neurons spike simultaneously
            | (next_times == next_internal_time),
            True, spike_mask)
        spike_mask = jnp.where(
            no_event | ~spike_in_layer,
            jnp.zeros_like(current_state.spike_times, dtype=bool),
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
        jnp.any(state.spike_mask),
        return_existing_event,
        find_new_events,
        state)
