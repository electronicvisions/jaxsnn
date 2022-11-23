from typing import Any, Callable, Tuple

import jax
import jax.numpy as np
from jax.tree_util import tree_flatten, tree_unflatten
from tree_math import Vector

from jaxsnn.base.types import Array, Spike, StepState


def exponential_flow(A: Array):
    def flow(x0: Array, t: float):
        return np.dot(jax.scipy.linalg.expm(A * t), x0)  # type: ignore

    def wrapped(x0: Vector, t: float) -> Vector:
        values, tree_def = tree_flatten(x0)
        res = flow(np.stack(values), t)
        return tree_unflatten(tree_def, res)

    return wrapped


def f(A, x0, t):
    return np.dot(jax.scipy.linalg.expm(A * t), x0)  # type: ignore


def jump_condition(dynamics, v_th, x0, t):
    return dynamics(x0, t)[0] - v_th  # this implements the P y(t) - b above


def step(
    dynamics: Callable,
    solver: Callable,
    tr_dynamics: Callable,
    t_max: float,
    input: Tuple[StepState, Tuple[Array, Array]],
    *args: int,
) -> Tuple[Tuple[StepState, Tuple[Array, Array]], Spike]:
    """Determine the next spike (external or internal), and integrate the neurons to that point.

    Args:
        dynamics (Callable): Function describing neuron dynamics
        solver (Callable): Parallel root solver
        tr_dynamics (Callable): function describing the transition dynamics
        t_max (float): Max time until which to run
        weights (Tuple[Array, Array]): input and recurrent weights
        input_spikes (Spike): input spikes (time and index)
        state (StepState): (Neuron state, current_time, running_idx)

    Returns:
        Tuple[StepState, Spike]: New state after transition and spike for storing
    """
    # TODO should solver only see StepState?
    state, weights = input
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
        spike_idx,
        recurrent_spike,
    )
    return (transitioned_state, weights), Spike(t_dyn, stored_idx)


def trajectory(dynamics, n_spikes) -> Callable[[Any, Any, Any], Spike]:
    def fun(initial_state, weights, input_spikes) -> Spike:
        s = StepState(
            neuron_state=initial_state,
            time=0.0,
            running_idx=0,
            input_spikes=input_spikes,
        )
        _, spikes = jax.lax.scan(dynamics, (s, weights), np.arange(n_spikes))  # type: ignore
        return spikes

    return fun
