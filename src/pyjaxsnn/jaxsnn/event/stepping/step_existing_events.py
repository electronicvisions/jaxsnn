from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jaxsnn.event.types import EventPropSpike, Solver, StepState
from jaxsnn.event.stepping.types import StepInput


def step_existing(  # pylint: disable=unused-argument,too-many-locals
    dynamics: Callable,
    tr_dynamics: Callable,
    t_max: float,
    event_stepper: Solver,
    step_input: StepInput,
    *args: int,
) -> Tuple[StepInput, EventPropSpike]:
    """
    Find next spike (external or internal), and simulate to that point.

    :param dynamics: Function describing the continuous neuron dynamics
    :param tr_dynamics: Function describing the transition dynamics
    :param t_max: Max time until which to run
    :param solver: Parallel root solver which returns the next event
    :param state: (StepState, weights, int)

    :returns: New state after transition and stored spike
    """
    state, weights, layer_start = step_input
    prev_layer_start = layer_start - weights.input.shape[0]

    empty_event = EventPropSpike(
        time=jnp.array(jnp.inf), idx=jnp.array(-1), current=jnp.array(0.))

    next_event, state = jax.lax.cond(
        state.input_queue.is_empty,
        lambda s: (empty_event, s),
        lambda s: jax.lax.cond(
            s.input_queue.peek().idx >= layer_start,
            lambda s: (s.input_queue.pop(), s),
            lambda s: (s.input_queue.peek(), s),
            s),
        state)

    spike_in_layer = next_event.idx >= layer_start

    # New neuron state
    evolved_neuron_state = dynamics(
        state.neuron_state, jnp.minimum(next_event.time, t_max) - state.time)

    # Create new event
    current = jax.lax.cond(
        spike_in_layer,
        lambda: evolved_neuron_state.I[next_event.idx - layer_start],
        lambda: next_event.current)

    new_event = EventPropSpike(next_event.time, next_event.idx, current)

    # Update step state
    evolved_state = StepState(
        neuron_state=evolved_neuron_state,
        spike_times=state.spike_times,
        spike_mask=state.spike_mask,
        time=jnp.minimum(next_event.time, t_max),
        input_queue=state.input_queue)

    # Transition state
    transitioned_state = jax.lax.cond(
        next_event.idx == -1,
        lambda *args: evolved_state,
        tr_dynamics,
        evolved_state,
        weights,
        jnp.zeros_like(state.spike_mask).at[
            next_event.idx - layer_start].set(True),
        spike_in_layer,
        prev_layer_start)

    return (transitioned_state, weights, layer_start), new_event
