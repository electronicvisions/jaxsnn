from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as np

from jaxsnn.types import Array, Spike, StepState


def f(A, x0, t):
    return np.dot(jax.scipy.linalg.expm(A * t), x0)  # type: ignore


def jump_condition(dynamics, v_th, x0, t):
    return dynamics(x0, t)[0] - v_th  # this implements the P y(t) - b above


def recurrent_transition(
    state: StepState, weights: Tuple[Array, Array], input_spikes: Spike, spike_idx: int
):
    _, recurrent_weights = weights
    y_minus, t, n_input_received = state
    tr_row = recurrent_weights[spike_idx]

    y_minus = y_minus.at[:, 1].set(y_minus[:, 1] + tr_row)
    y_minus = y_minus.at[spike_idx, 0].set(0.0)
    return StepState(y_minus, t, n_input_received)


def input_transition(
    state: StepState, weights: Tuple[Array, Array], input_spikes: Spike, spike_idx: int
):
    input_weights, _ = weights
    y_minus, t, n_input_received = state
    tr_row = input_weights[input_spikes.idx[n_input_received]]
    y_minus = y_minus.at[:, 1].set(y_minus[:, 1] + tr_row)
    return StepState(y_minus, t, n_input_received + 1)


def transition(
    state: StepState,
    weights: Tuple[Array, Array],
    input_spikes: Spike,
    spike_idx: int,
    recurrent_spike: bool,
) -> StepState:
    return jax.lax.cond(
        recurrent_spike,
        recurrent_transition,
        input_transition,
        state,
        weights,
        input_spikes,
        spike_idx,
    )


def transition_without_recurrence(
    state: StepState,
    weights: Array,
    input_spikes: Spike,
    spike_idx: int,
    recurrent_spike: bool,
) -> StepState:
    def input_transition(
        state: StepState, weights: Array, input_spikes: Spike, spike_idx: int
    ):
        y_minus, t, n_input_received = state
        tr_row = weights[input_spikes.idx[n_input_received]]
        y_minus = y_minus.at[:, 1].set(y_minus[:, 1] + tr_row)
        return StepState(y_minus, t, n_input_received + 1)

    def no_transition(state: StepState, *args):
        y_minus, t, n_input_received = state
        y_minus = y_minus.at[spike_idx, 0].set(0.0)
        return StepState(y_minus, t, n_input_received)

    return jax.lax.cond(
        recurrent_spike,
        no_transition,
        input_transition,
        state,
        weights,
        input_spikes,
        spike_idx,
    )


def step(
    dynamics: Callable,
    solver: Callable,
    tr_dynamics: Callable,
    t_max: float,
    weights: Tuple[Array, Array],
    input_spikes: Spike,
    state: StepState,
    *args: int,
) -> Tuple[StepState, Spike]:
    """Determine the next spike (external or internal), and integrate the neurons to that point.

    Args:
        dynamics (Callable): Function describing neuron dynamics
        solver (Callable): Parallel root solver
        tr_dynamics (Callable): function describing the tra
        t_max (float): Max time until which to run
        weights (Tuple[Array, Array]): input and recurrent weights
        input_spikes (Spike): input spikes (time and index)
        state (StepState): (Neuron state, current_time, n_input_received)

    Returns:
        Tuple[StepState, Spike]: New state after transition and spike for storing
    """
    y, t, n_input_received = state
    pred_spikes = solver(y, t_max - t) + t
    spike_idx = np.argmin(pred_spikes)

    # integrate state
    t_dyn = np.min(
        np.array([pred_spikes[spike_idx], input_spikes.time[n_input_received], t_max])
    )
    state = StepState(dynamics(y, t_dyn - t), t_dyn, n_input_received)

    # determine spike nature
    no_spike = t_dyn == t_max
    recurrent_spike = pred_spikes[spike_idx] < input_spikes.time[n_input_received]

    stored_idx = jax.lax.cond(recurrent_spike, lambda: spike_idx, lambda: -1)
    transitioned_state = jax.lax.cond(
        no_spike,
        lambda *args: state,
        tr_dynamics,
        state,
        weights,
        input_spikes,
        spike_idx,
        recurrent_spike,
    )
    return transitioned_state, Spike(t_dyn, stored_idx)


def forward_integration(step_fn, n_spikes, weights, input_spikes) -> Spike:
    n_hidden = weights[1].shape[0]
    state = StepState(np.zeros((n_hidden, 2)), 0, 0)
    _, spikes = jax.lax.scan(partial(step_fn, weights, input_spikes), state, np.arange(n_spikes))  # type: ignore
    return spikes
