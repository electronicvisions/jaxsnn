from jaxsnn.functional.leaky_integrate_and_fire import LIFParameters, LIFState
from jaxsnn.event.functional import (
    forward_integration,
    step,
    exponential_flow,
)
from jaxsnn.types import Array, Spike, Weight, StepState

import jax
import jax.numpy as np

from functools import partial
from typing import Tuple


def lif_exponential_flow(p: LIFParameters):
    A = np.array([[-p.tau_mem_inv, p.tau_mem_inv], [0, -p.tau_syn_inv]])
    return exponential_flow(A)


def jump_condition(dynamics, v_th, x0, t):
    return dynamics(x0, t)[0] - v_th


def recurrent_transition(
    state: StepState, weights: Tuple[Array, Array], _: Spike, spike_idx: int
):
    _, recurrent_weights = weights
    y_minus, t, n_input_received = state
    tr_row = recurrent_weights[spike_idx]

    y_minus = y_minus.at[:, 1].set(y_minus[:, 1] + tr_row)
    y_minus = y_minus.at[spike_idx, 0].set(0.0)
    return StepState(y_minus, t, n_input_received)


def input_transition(
    state: StepState, weights: Tuple[Array, Array], input_spikes: Spike, _: int
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
    def input_transition(state: StepState, weights: Array, input_spikes: Spike, _: int):
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


def RecursiveLIF(n_hidden: int, n_spikes: int, t_max: float, p: LIFParameters, solver):

    single_flow = lif_exponential_flow(p)
    dynamics = jax.vmap(single_flow, in_axes=(0, None))

    # solver = partial(solver, 1 / p.tau_mem_inv, p.v_th)
    batched_solver = jax.vmap(solver, in_axes=(0, None))

    step_fn = partial(step, dynamics, batched_solver, transition, t_max)
    forward = partial(forward_integration, step_fn, n_spikes)

    def init_fn(rng: jax.random.KeyArray, input_shape: int) -> Weight:
        scale_factor = 2.0
        input_rng, hidden_rng = jax.random.split(rng)
        input_weights = (
            jax.random.uniform(input_rng, (input_shape, n_hidden)) * scale_factor
        )
        rec_weights = (
            jax.random.uniform(hidden_rng, (n_hidden, n_hidden)) * scale_factor
        )
        return n_hidden, (input_weights, rec_weights)

    return init_fn, forward


def LIF(n_hidden: int, n_spikes: int, t_max: float, p: LIFParameters, solver):
    # TODO: In common with RecurrentLIF, have to abstract over "ttf-solver" aswell
    single_flow = lif_exponential_flow(p)
    dynamics = jax.vmap(single_flow, in_axes=(0, None))
    # solver = partial(ttfs_solver, 1 / p.tau_mem_inv, p.v_th)
    batched_solver = jax.vmap(solver, in_axes=(0, None))

    non_recursive_step_fn = partial(
        step, dynamics, batched_solver, transition_without_recurrence, t_max
    )
    forward = partial(forward_integration, non_recursive_step_fn, n_spikes)

    def init_fn(rng: jax.random.KeyArray, input_shape: int):
        scale_factor = 2.0
        return n_hidden, jax.random.uniform(rng, (input_shape, n_hidden)) * scale_factor

    return init_fn, forward
