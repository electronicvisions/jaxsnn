from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as np

from jaxsnn.types import Array, Spike, StepState

from typing import Any


def exponential_flow(A):
    def flow(x0, t):
        return np.dot(jax.scipy.linalg.expm(A * t), x0)  # type: ignore

    return flow


def step(
    dynamics: Callable,
    solver: Callable,
    tr_dynamics: Callable,
    t_max: float,
    # all together
    weights: Tuple[Array, Array],
    state: StepState,
    *args: int,
) -> Tuple[StepState, Spike]:
    """Determine the next spike (external or internal), and integrate the neurons to that point.

    Args:
        dynamics (Callable): Function describing neuron dynamics
        solver (Callable): Parallel root solver
        tr_dynamics (Callable): function describing the transition dynamics
        t_max (float): Max time until which to run
        weights (Tuple[Array, Array]): input and recurrent weights
        input_spikes (Spike): input spikes (time and index)
        state (StepState): (Neuron state, current_time, n_input_received)

    Returns:
        Tuple[StepState, Spike]: New state after transition and spike for storing
    """
    # y, t, n_input_received = state
    pred_spikes = solver(state.neuron_state, t_max - state.time) + state.time
    spike_idx = np.argmin(pred_spikes)

    # integrate state
    t_dyn = np.min(
        np.array(
            [
                pred_spikes[spike_idx],
                state.input_spikes.time[state.running_idx],
                t_max,
            ]
        )
    )
    state = StepState(
        neuron_state=dynamics(state.neuron_state, t_dyn - state.time),
        time=t_dyn,
        running_idx=state.running_idx,
        input_spikes=state.input_spikes,
    )

    # determine spike nature
    no_spike = t_dyn == t_max
    recurrent_spike = (
        pred_spikes[spike_idx] < state.input_spikes.time[state.running_idx]
    )

    stored_idx = jax.lax.cond(recurrent_spike, lambda: spike_idx, lambda: -1)
    transitioned_state = jax.lax.cond(
        no_spike,
        lambda *args: state,
        tr_dynamics,
        state,
        weights,
        state.input_spikes,
        spike_idx,
        recurrent_spike,
    )
    return transitioned_state, Spike(t_dyn, stored_idx)


def forward_integration(step_fn, n_spikes, weights, input_spikes) -> Spike:
    # TODO: This makes assumptions on the form of the initial state, instead
    #       this should conform to the API defined in jaxsnn.base.funcutils
    n_hidden = weights[1].shape[0]
    state = StepState(
        neuron_state=np.zeros((n_hidden, 2)),
        time=0.0,
        running_idx=0,
        input_spikes=input_spikes,
    )
    _, spikes = jax.lax.scan(partial(step_fn, weights), state, np.arange(n_spikes))  # type: ignore
    return spikes


def trajectory(dynamics, n_spikes) -> Callable[[Any, Any], Spike]:
    # dynamcis = partial(step_fn, weights)
    def fun(initial_state, input_spikes) -> Spike:
        s = StepState(
            neuron_state=initial_state,
            time=0.0,
            running_idx=0,
            input_spikes=input_spikes,
        )
        _, spikes = jax.lax.scan(dynamics, s, np.arange(n_spikes))  # type: ignore
        return spikes

    return fun
