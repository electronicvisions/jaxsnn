from typing import Callable, Tuple, Union

import jax
import jax.numpy as np
from jaxsnn.event.types import (
    EventPropSpike,
    HWLossFn,
    InputQueue,
    LIFState,
    LossFn,
    SingleApply,
    Solver,
    StepState,
    Weight,
)


def batch_wrapper(
    loss_fn: Union[LossFn, HWLossFn],
    in_axes: tuple = (None, 0),
    pmap: bool = False,
):
    """Add an outer batch dimension to `loss_fn`.

    The loss function returns the actual loss value, and some more information.
    When adding the batch dimension, the average of the loss value is taken,
    bu the information is stacked.
    """

    def wrapped_fn(*args, **kwargs):
        if pmap:
            res = jax.pmap(loss_fn, in_axes=in_axes)(*args, **kwargs)
        else:
            res = jax.vmap(loss_fn, in_axes=in_axes)(*args, **kwargs)
        return np.mean(res[0]), res[1]

    return wrapped_fn


# Input to step function.
# Consists of StepState, weights of the network and start index of the layer
StepInput = Tuple[StepState, Weight, int]


def step(  # pylint: disable=unused-argument,too-many-locals
    dynamics: Callable,
    tr_dynamics: Callable,
    solver: Solver,
    t_max: float,
    step_input: StepInput,
    *args: int,
) -> Tuple[StepInput, EventPropSpike]:
    """Find next spike (external or internal), and simulate to that point.

    Args:
        dynamics (Callable): Function describing the continous neuron dynamics
        tr_dynamics (Callable): Function describing the transition dynamics
        t_max (float): Max time until which to run
        solver (Solver): Parallel root solver which returns the next event
        state (StepInput): (StepState, weights, int)
    Returns:
        Tuple[StepInput, Spike]: New state after transition and stored spike
    """
    state, weights, layer_start = step_input
    prev_layer_start = layer_start - weights.input.shape[0]

    next_internal = solver(state.neuron_state, state.time, t_max)

    # determine spike nature and spike time
    input_time = jax.lax.cond(
        state.input_queue.is_empty,
        lambda: t_max,
        lambda: state.input_queue.peek().time,
    )
    t_dyn = np.minimum(next_internal.time, input_time)

    # comparing only makes sense if exactly dt is returned from solver
    spike_in_layer = next_internal.time < input_time
    no_event = t_dyn + 1e-6 > t_max
    stored_idx = jax.lax.cond(
        no_event,
        lambda: -1,
        lambda: jax.lax.cond(
            spike_in_layer,
            lambda: next_internal.idx + layer_start,
            lambda: state.input_queue.peek().idx,
        ),
    )
    state = StepState(
        neuron_state=dynamics(state.neuron_state, t_dyn - state.time),
        time=t_dyn,
        input_queue=state.input_queue,
    )
    current = jax.lax.cond(
        spike_in_layer,
        lambda: state.neuron_state.I[next_internal.idx],
        lambda: state.input_queue.peek().current,
    )
    transitioned_state = jax.lax.cond(
        no_event,
        lambda *args: state,
        tr_dynamics,
        state,
        weights,
        next_internal.idx,
        spike_in_layer,
        prev_layer_start,
    )
    return (transitioned_state, weights, layer_start), EventPropSpike(
        t_dyn, stored_idx, current
    )


def trajectory(
    step_fn: Callable[[StepInput, int], Tuple[StepInput, EventPropSpike]],
    n_hidden: int,
    n_spikes: int,
) -> SingleApply:
    """Evaluate the `step_fn` until `n_spikes` have been simulated.

    Uses a scan over the `step_fn` to return an apply function
    """

    def fun(
        layer_start: int,
        weights: Weight,
        input_spikes: EventPropSpike,
    ) -> EventPropSpike:
        initial_state = LIFState(np.zeros(n_hidden), np.zeros(n_hidden))
        step_state = StepState(
            neuron_state=initial_state,
            time=0.0,
            input_queue=InputQueue(input_spikes),
        )
        _, spikes = jax.lax.scan(
            step_fn, (step_state, weights, layer_start), np.arange(n_spikes)
        )
        return spikes

    return fun
