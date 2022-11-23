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
    state: StepState, weights: Tuple[Array, Array], spike_idx: int
):
    _, recurrent_weights = weights
    tr_row = recurrent_weights[spike_idx]

    state.neuron_state.I = state.neuron_state.I + tr_row
    state.neuron_state.V = state.neuron_state.V.at[spike_idx].set(0.0)
    return state


def input_transition(state: StepState, weights: Tuple[Array, Array], spike_idx: int):
    input_weights, _ = weights
    tr_row = input_weights[state.input_spikes.idx[state.running_idx]]
    state.neuron_state.I = state.neuron_state.I + tr_row
    state.running_idx += 1
    return state


def transition(
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
    )


def transition_without_recurrence(
    state: StepState,
    weights: Array,
    spike_idx: int,
    recurrent_spike: bool,
) -> StepState:
    def input_transition(state: StepState, weights: Array, spike_idx: int):
        tr_row = weights[state.input_spikes.idx[state.running_idx]]
        state.neuron_state.I = state.neuron_state.I + tr_row
        state.running_idx += 1
        return state

    def no_transition(state: StepState, *args):
        state.neuron_state.V = state.neuron_state.V.at[spike_idx].set(0, 0)
        return state

    return jax.lax.cond(
        recurrent_spike,
        no_transition,
        input_transition,
        state,
        weights,
        spike_idx,
    )


def RecursiveLIF(n_hidden: int, n_spikes: int, t_max: float, p: LIFParameters, solver):

    single_flow = lif_exponential_flow(p)
    dynamics = jax.vmap(single_flow, in_axes=(0, None))
    batched_solver = jax.vmap(solver, in_axes=(0, None))

    step_fn = partial(step, dynamics, batched_solver, transition, t_max)
    forward = trajectory(step_fn, n_spikes)
    initial_state = LIFState(np.zeros(n_hidden), np.zeros(n_hidden))

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

    return init_fn, partial(forward, initial_state)


def LIF(n_hidden: int, n_spikes: int, t_max: float, p: LIFParameters, solver):
    # TODO: In common with RecurrentLIF, have to abstract over "ttf-solver" aswell
    single_flow = lif_exponential_flow(p)
    dynamics = jax.vmap(single_flow, in_axes=(0, None))
    batched_solver = jax.vmap(solver, in_axes=(0, None))

    step_fn = partial(
        step, dynamics, batched_solver, transition_without_recurrence, t_max
    )
    forward = trajectory(step_fn, n_spikes)
    initial_state = LIFState(np.zeros(n_hidden), np.zeros(n_hidden))

    def init_fn(rng: jax.random.KeyArray, input_shape: int):
        scale_factor = 2.0
        return n_hidden, jax.random.uniform(rng, (input_shape, n_hidden)) * scale_factor

    return init_fn, partial(forward, initial_state)
