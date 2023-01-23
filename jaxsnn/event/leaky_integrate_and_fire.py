from functools import partial
from typing import Tuple

import jax
import jax.numpy as np

from jaxsnn.base.types import Array, StepState, Weight
from jaxsnn.event.functional import exponential_flow, step, trajectory
from jaxsnn.functional.leaky_integrate_and_fire import LIFParameters, LIFState


def lif_exponential_flow(p: LIFParameters):
    A = np.array([[-p.tau_mem_inv, p.tau_mem_inv], [0, -p.tau_syn_inv]])
    return exponential_flow(A)


def jump_condition(dynamics, v_th, x0, t):
    return dynamics(x0, t)[0] - v_th


def recurrent_transition(
    state: StepState, weights: Tuple[Array, Array], spike_idx: int, v_reset: float
):
    _, recurrent_weights = weights
    tr_row = recurrent_weights[spike_idx]

    state.neuron_state.I = state.neuron_state.I + tr_row
    state.neuron_state.V = state.neuron_state.V.at[spike_idx].set(v_reset)
    return state


def input_transition(
    state: StepState, weights: Tuple[Array, Array], spike_idx: int, v_reset: float
):
    input_weights, _ = weights
    input_previous_layer = state.input_spikes.idx[state.running_idx] < 0
    tr_row = input_weights[state.input_spikes.idx[state.running_idx]]
    state.neuron_state.I = jax.lax.cond(
        input_previous_layer,
        lambda: state.neuron_state.I,
        lambda: state.neuron_state.I + tr_row,
    )
    state.running_idx += 1
    return state


def transition_with_reccurence(
    v_reset: float,
    state: StepState,
    weights: Tuple[Array, Array],
    spike_idx: int,
    recurrent_spike: bool,
) -> StepState:
    return jax.lax.cond(
        recurrent_spike,
        recurrent_transition,
        input_transition,
        state,
        weights,
        spike_idx,
        v_reset,
    )


def transition_without_recurrence(
    v_reset: float,
    state: StepState,
    weights: Array,
    spike_idx: int,
    recurrent_spike: bool,
) -> StepState:
    def input_transition(state: StepState, weights: Array, spike_idx: int):
        input_previous_layer = state.input_spikes.idx[state.running_idx] < 0
        tr_row = weights[state.input_spikes.idx[state.running_idx]]
        state.neuron_state.I = jax.lax.cond(
            input_previous_layer,
            lambda: state.neuron_state.I,
            lambda: state.neuron_state.I + tr_row,
        )
        # state.neuron_state.I = state.neuron_state.I + tr_row
        state.running_idx += 1
        return state

    def no_transition(state: StepState, *args):
        state.neuron_state.V = state.neuron_state.V.at[spike_idx].set(v_reset)
        return state

    return jax.lax.cond(
        recurrent_spike,
        no_transition,
        input_transition,
        state,
        weights,
        spike_idx,
    )


def RecursiveLIF(
    n_hidden: int,
    n_spikes: int,
    t_max: float,
    p: LIFParameters,
    solver,
    mean=0.5,
    std=2.0,
):
    single_flow = lif_exponential_flow(p)
    dynamics = jax.vmap(single_flow, in_axes=(0, None))
    batched_solver = jax.vmap(solver, in_axes=(0, None))
    transition = partial(transition_with_reccurence, p.v_reset)

    step_fn = partial(step, dynamics, batched_solver, transition, t_max)
    forward = trajectory(step_fn, n_spikes)
    initial_state = LIFState(np.zeros(n_hidden), np.zeros(n_hidden))

    def init_fn(rng: jax.random.KeyArray, input_shape: int) -> Weight:
        input_rng, hidden_rng = jax.random.split(rng)
        input_weights = (
            jax.random.normal(input_rng, (input_shape, n_hidden)) * std + mean
        )
        rec_weights = jax.random.normal(hidden_rng, (n_hidden, n_hidden)) * std + mean
        return n_hidden, (input_weights, rec_weights)

    return init_fn, partial(forward, initial_state)


def LIF(
    n_hidden: int,
    n_spikes: int,
    t_max: float,
    p: LIFParameters,
    solver,
    mean=0.5,
    std=2.0,
):
    # TODO: In common with RecurrentLIF, have to abstract over "ttf-solver" aswell
    single_flow = lif_exponential_flow(p)
    dynamics = jax.vmap(single_flow, in_axes=(0, None))
    batched_solver = jax.vmap(solver, in_axes=(0, None))
    transition = partial(transition_without_recurrence, p.v_reset)

    step_fn = partial(step, dynamics, batched_solver, transition, t_max)
    forward = trajectory(step_fn, n_spikes)
    initial_state = LIFState(np.zeros(n_hidden), np.zeros(n_hidden))

    def init_fn(rng: jax.random.KeyArray, input_shape: int):
        return n_hidden, jax.random.normal(rng, (input_shape, n_hidden)) * std + mean

    return init_fn, partial(forward, initial_state)
