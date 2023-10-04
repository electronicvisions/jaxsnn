from functools import partial

import jax
import jax.numpy as np
from typing import List, Tuple, Optional

from jaxsnn.base.types import (
    Array,
    WeightRecurrent,
    WeightInput,
    StepState,
    InputQueue,
    Spike,
    EventPropSpike,
)
from jaxsnn.event.adjoint_lif import (
    adjoint_lif_exponential_flow,
    adjoint_transition_without_recurrence,
    adjoint_transition_with_recurrence,
)
from jaxsnn.event.functional import exponential_flow, step, trajectory
from jaxsnn.functional.leaky_integrate_and_fire import LIFParameters, LIFState
from jaxsnn.event.adjoint_lif import step_bwd
from jaxsnn.event.transition import (
    transition_with_recurrence,
    transition_without_recurrence,
)
from jaxsnn.event.root.next import next_event, next_queue


def lif_exponential_flow(p: LIFParameters):
    A = np.array([[-p.tau_mem_inv, p.tau_mem_inv], [0, -p.tau_syn_inv]])
    return exponential_flow(A)


def lif_dynamics(p: LIFParameters, x0: Array, t: float):
    tau_exp = np.exp(-t / p.tau_mem)
    syn_exp = np.exp(-t / p.tau_syn)
    A = np.array(
        [
            [tau_exp, p.tau_syn / (p.tau_mem - p.tau_syn) * (tau_exp - syn_exp)],
            [0, syn_exp],
        ]
    )
    return np.dot(A, x0)


def LIF(
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
    batched_solver = partial(next_event, jax.vmap(solver, in_axes=(0, None)))

    transition = partial(transition_without_recurrence, p)
    step_fn = partial(step, dynamics, transition, t_max, batched_solver)

    initial_state = LIFState(np.zeros(n_hidden), np.zeros(n_hidden))
    forward = trajectory(step_fn, n_spikes)

    def init_fn(rng: jax.random.KeyArray, input_shape: int):
        return n_hidden, WeightInput(
            jax.random.normal(rng, (input_shape, n_hidden)) * std + mean
        )

    return init_fn, partial(forward, initial_state)


def RecurrentLIF(
    layers: List[int],
    n_spikes: int,
    t_max: float,
    p: LIFParameters,
    solver,
    mean: List[float],
    std: List[float],
):
    single_flow = lif_exponential_flow(p)
    dynamics = jax.vmap(single_flow, in_axes=(0, None))
    batched_solver = partial(next_event, jax.vmap(solver, in_axes=(0, None)))
    transition = partial(transition_with_recurrence, p)

    step_fn = partial(step, dynamics, transition, t_max, batched_solver)

    hidden_size = np.sum(np.array(layers))
    initial_state = LIFState(np.zeros(hidden_size), np.zeros(hidden_size))

    forward = trajectory(step_fn, n_spikes)

    def init_fn(
        rng: jax.random.KeyArray, input_size: int
    ) -> Tuple[int, WeightRecurrent]:
        assert len(layers) >= 1

        rng, layer_rng = jax.random.split(rng)
        input_weights = (
            jax.random.normal(layer_rng, (input_size, layers[0])) * std[0] + mean[0]
        )
        input_weights = (
            np.zeros((input_size, hidden_size)).at[:, : layers[0]].set(input_weights)
        )

        recurrent_weights = np.zeros((hidden_size, hidden_size))
        l_sum = 0
        for i, (l1, l2) in enumerate(zip(layers, layers[1:])):
            rng, layer_rng = jax.random.split(rng)
            recurrent_weights = recurrent_weights.at[
                l_sum : l_sum + l1, l_sum + l1 : l_sum + l1 + l2
            ].set(jax.random.normal(layer_rng, (l1, l2)) * std[i + 1] + mean[i + 1])
            l_sum += l1

        weights = WeightRecurrent(input_weights, recurrent_weights)
        return hidden_size, weights

    return init_fn, partial(forward, initial_state)


def EventPropLIF(
    n_hidden: int,
    n_spikes: int,
    t_max: float,
    p: LIFParameters,
    solver,
    mean=0.5,
    std=2.0,
    duplication: Optional[int] = None,
):
    single_flow = lif_exponential_flow(p)
    dynamics = jax.vmap(single_flow, in_axes=(0, None))
    batched_solver = partial(next_event, jax.vmap(solver, in_axes=(0, None)))
    transition = partial(transition_without_recurrence, p)

    single_adjoint_flow = adjoint_lif_exponential_flow(p)
    adjoint_dynamics = jax.vmap(single_adjoint_flow, in_axes=(0, None))
    adjoint_tr_dynamics = partial(adjoint_transition_without_recurrence, p)

    step_fn = partial(step, dynamics, transition, t_max, batched_solver)
    step_fn_bwd = partial(step_bwd, adjoint_dynamics, adjoint_tr_dynamics, t_max)

    # define custom forward to save data
    def step_fn_fwd(input, iteration: int):
        ((state, weights, layer_start), spike) = step_fn(input, iteration)
        return ((state, weights, layer_start), spike), (
            spike,
            weights,
            layer_start,
            state.input_queue.head,
        )

    # define custom vjp
    step_fn_event_prop = jax.custom_vjp(step_fn)
    step_fn_event_prop.defvjp(step_fn_fwd, step_fn_bwd)

    forward = trajectory(step_fn_event_prop, n_spikes)
    initial_state = LIFState(np.zeros(n_hidden), np.zeros(n_hidden))

    def init_fn(rng: jax.random.KeyArray, input_shape: int) -> Tuple[int, WeightInput]:
        if duplication is not None:
            weights = jax.random.normal(rng, (int(input_shape / duplication), n_hidden))
            weights = np.repeat(weights, duplication, axis=0)
        else:
            weights = jax.random.normal(rng, (input_shape, n_hidden))
        return n_hidden, WeightInput(weights * std + mean)

    return init_fn, partial(forward, initial_state)


def RecurrentEventPropLIF(
    layers: List[int],
    n_spikes: int,
    t_max: float,
    p: LIFParameters,
    solver,
    mean: List[float],
    std: List[float],
    wrap_only_step: bool = False,
):
    single_flow = lif_exponential_flow(p)
    dynamics = jax.vmap(single_flow, in_axes=(0, None))
    batched_solver = partial(next_event, jax.vmap(solver, in_axes=(0, None)))
    transition = partial(transition_with_recurrence, p)

    single_adjoint_flow = adjoint_lif_exponential_flow(p)
    adjoint_dynamics = jax.vmap(single_adjoint_flow, in_axes=(0, None))
    adjoint_tr_dynamics = partial(adjoint_transition_with_recurrence, p)

    step_fn = partial(step, dynamics, transition, t_max, batched_solver)
    step_fn_bwd = partial(step_bwd, adjoint_dynamics, adjoint_tr_dynamics, t_max)

    hidden_size = np.sum(np.array(layers))
    initial_state = LIFState(np.zeros(hidden_size), np.zeros(hidden_size))

    def init_fn(
        rng: jax.random.KeyArray, input_size: int
    ) -> Tuple[int, WeightRecurrent]:
        assert len(layers) >= 1

        rng, layer_rng = jax.random.split(rng)
        input_weights = (
            jax.random.normal(layer_rng, (input_size, layers[0])) * std[0] + mean[0]
        )
        input_weights = (
            np.zeros((input_size, hidden_size)).at[:, : layers[0]].set(input_weights)
        )

        recurrent_weights = np.zeros((hidden_size, hidden_size))
        l_sum = 0
        for i, (l1, l2) in enumerate(zip(layers, layers[1:])):
            rng, layer_rng = jax.random.split(rng)
            recurrent_weights = recurrent_weights.at[
                l_sum : l_sum + l1, l_sum + l1 : l_sum + l1 + l2
            ].set(jax.random.normal(layer_rng, (l1, l2)) * std[i + 1] + mean[i + 1])
            l_sum += l1

        weights = WeightRecurrent(input_weights, recurrent_weights)
        return hidden_size, weights

    if wrap_only_step:
        # define custom forward to save data
        def step_fn_fwd(input, iteration: int):
            ((state, weights, layer_start), spike) = jax.jit(step_fn)(input, iteration)
            return ((state, weights, layer_start), spike), (
                spike,
                weights,
                layer_start,
                state.input_queue.head,
            )

        # define custom vjp only for the step function
        step_fn_event_prop = jax.custom_vjp(step_fn)
        step_fn_event_prop.defvjp(step_fn_fwd, step_fn_bwd)
        return init_fn, partial(trajectory(step_fn_event_prop, n_spikes), initial_state)

    # wrap step bwd so it is compliant with scan syntax
    def step_bwd_wrapper(weights, init, xs):
        adjoint_state, grads, layer_start = init
        spike, adjoint_spike = xs
        res = (
            spike,
            weights,
            layer_start,
            len(adjoint_state.input_queue.spikes.time) - adjoint_state.input_queue.head,
        )
        g = (adjoint_state, grads, 0), adjoint_spike
        return step_fn_bwd(res, g)

    def custom_trajectory(s, weights: int, layer_start: int):
        adjoint_state, spikes = jax.lax.scan(
            step_fn, (s, weights, layer_start), np.arange(n_spikes)
        )
        return adjoint_state, spikes

    def custom_trajectory_fwd(s, weights, layer_start: int):
        # here the hardware call can be injected instead of calling forward
        output_state, spikes = custom_trajectory(s, weights, layer_start)
        return (output_state, spikes), (spikes, output_state)

    def custom_trajectory_bwd(res, g):
        spikes, (state, weights, layer_start) = res
        (adjoint_state, grads, _), adjoint_spikes = g

        (adjoint_state, grads, layer_start), _ = jax.lax.scan(
            partial(step_bwd_wrapper, weights),
            (adjoint_state, grads, layer_start),
            (spikes, adjoint_spikes),
            reverse=True,
        )
        return adjoint_state, grads, 0

    custom_trajectory = jax.custom_vjp(custom_trajectory)
    custom_trajectory.defvjp(custom_trajectory_fwd, custom_trajectory_bwd)

    def apply_fn(layer_start: int, weights, input_spikes):
        s = StepState(
            neuron_state=initial_state,
            time=0.0,
            input_queue=InputQueue(input_spikes),
        )
        _, spikes = custom_trajectory(s, weights, layer_start)
        return spikes

    return init_fn, apply_fn


def HardwareRecurrentLIF(
    layers: List[int],
    n_spikes: int,
    t_max: float,
    p: LIFParameters,
    mean: List[float],
    std: List[float],
    duplication: Optional[int] = None,
):
    single_flow = lif_exponential_flow(p)
    dynamics = jax.vmap(single_flow, in_axes=(0, None))
    transition = partial(transition_with_recurrence, p)

    single_adjoint_flow = adjoint_lif_exponential_flow(p)
    adjoint_dynamics = jax.vmap(single_adjoint_flow, in_axes=(0, None))
    adjoint_tr_dynamics = partial(adjoint_transition_with_recurrence, p)

    step_fn_no_solver = partial(step, dynamics, transition, t_max)
    step_fn_bwd = partial(step_bwd, adjoint_dynamics, adjoint_tr_dynamics, t_max)

    hidden_size = np.sum(np.array(layers))
    initial_state = LIFState(np.zeros(hidden_size), np.zeros(hidden_size))

    def init_fn(
        rng: jax.random.KeyArray, input_size: int
    ) -> Tuple[int, WeightRecurrent]:
        assert len(layers) >= 1

        rng, layer_rng = jax.random.split(rng)
        if duplication is not None:
            input_weights = jax.random.normal(
                layer_rng, (int(input_size / duplication), layers[0])
            )
            input_weights = np.repeat(input_weights, duplication, axis=0)
        else:
            input_weights = jax.random.normal(layer_rng, (input_size, layers[0]))
        input_weights = (
            np.zeros((input_size, hidden_size))
            .at[:, : layers[0]]
            .set(input_weights * std[0] + mean[0])
        )

        recurrent_weights = np.zeros((hidden_size, hidden_size))
        l_sum = 0
        for i, (l1, l2) in enumerate(zip(layers, layers[1:])):
            rng, layer_rng = jax.random.split(rng)
            recurrent_weights = recurrent_weights.at[
                l_sum : l_sum + l1, l_sum + l1 : l_sum + l1 + l2
            ].set(jax.random.normal(layer_rng, (l1, l2)) * std[i + 1] + mean[i + 1])
            l_sum += l1

        weights = WeightRecurrent(input_weights, recurrent_weights)
        return hidden_size, weights

    # wrap step bwd so it is compliant with scan syntax
    def step_bwd_wrapper(weights, init, xs):
        adjoint_state, grads, layer_start = init
        spike, adjoint_spike = xs
        res = (
            spike,
            weights,
            layer_start,
            len(adjoint_state.input_queue.spikes.time) - adjoint_state.input_queue.head,
        )
        g = (adjoint_state, grads, 0), adjoint_spike
        return step_fn_bwd(res, g)

    def custom_trajectory(
        s: StepState,
        weights: List[WeightRecurrent],
        layer_start: int,
        known_spikes: Spike,
    ):
        # attach known spikes to "root solver"
        step_fn = partial(
            step_fn_no_solver, partial(next_queue, known_spikes, layer_start)
        )
        state, spikes = jax.lax.scan(
            step_fn, (s, weights, layer_start), np.arange(n_spikes)
        )
        return state, spikes

    def custom_trajectory_fwd(
        s: StepState,
        weights: List[WeightRecurrent],
        layer_start: int,
        known_spikes: Spike,
    ):
        # here the hardware call can be injected instead of calling forward
        output_state, spikes = custom_trajectory(s, weights, layer_start, known_spikes)
        return (output_state, spikes), (spikes, output_state, known_spikes)

    def custom_trajectory_bwd(res, g):
        spikes, (state, weights, layer_start), known_spikes = res
        (adjoint_state, grads, _), adjoint_spikes = g

        (adjoint_state, grads, layer_start), _ = jax.lax.scan(
            partial(step_bwd_wrapper, weights),
            (adjoint_state, grads, layer_start),
            (spikes, adjoint_spikes),
            reverse=True,
        )
        return adjoint_state, grads, 0, known_spikes

    custom_trajectory = jax.custom_vjp(custom_trajectory)
    custom_trajectory.defvjp(custom_trajectory_fwd, custom_trajectory_bwd)

    def apply_fn(
        layer_start: int, weights, input_spikes: EventPropSpike, known_spikes: Spike
    ):
        s = StepState(
            neuron_state=initial_state,
            time=0.0,
            input_queue=InputQueue(input_spikes),
        )
        _, spikes = custom_trajectory(s, weights, layer_start, known_spikes)
        return spikes

    return init_fn, apply_fn


# TODO a lot of code duplication with Reccurrent Version
def HardwareLIF(
    n_hidden: int,
    n_spikes: int,
    t_max: float,
    p: LIFParameters,
    mean: float,
    std: float,
    duplication: Optional[int] = None,
):
    single_flow = lif_exponential_flow(p)
    dynamics = jax.vmap(single_flow, in_axes=(0, None))
    transition = partial(transition_without_recurrence, p)

    single_adjoint_flow = adjoint_lif_exponential_flow(p)
    adjoint_dynamics = jax.vmap(single_adjoint_flow, in_axes=(0, None))
    adjoint_tr_dynamics = partial(adjoint_transition_without_recurrence, p)

    step_fn_no_solver = partial(step, dynamics, transition, t_max)
    step_fn_bwd = partial(step_bwd, adjoint_dynamics, adjoint_tr_dynamics, t_max)

    initial_state = LIFState(np.zeros(n_hidden), np.zeros(n_hidden))

    def init_fn(rng: jax.random.KeyArray, input_size: int) -> Tuple[int, WeightInput]:

        rng, layer_rng = jax.random.split(rng)
        if duplication is not None:
            weights = jax.random.normal(
                layer_rng, (int(input_size / duplication), n_hidden)
            )
            weights = np.repeat(weights, duplication, axis=0)
        else:
            weights = jax.random.normal(layer_rng, (input_size, n_hidden))
        return n_hidden, WeightInput(weights * std + mean)

    # wrap step bwd so it is compliant with scan syntax
    def step_bwd_wrapper(weights, init, xs):
        adjoint_state, grads, layer_start = init
        spike, adjoint_spike = xs
        res = (
            spike,
            weights,
            layer_start,
            len(adjoint_state.input_queue.spikes.time) - adjoint_state.input_queue.head,
        )
        g = (adjoint_state, grads, 0), adjoint_spike
        return step_fn_bwd(res, g)

    def custom_trajectory(
        s: StepState,
        weights: List[WeightRecurrent],
        layer_start: int,
        known_spikes: Spike,
    ):
        # attach known spikes to "root solver"
        step_fn = partial(
            step_fn_no_solver, partial(next_queue, known_spikes, layer_start)
        )
        state, spikes = jax.lax.scan(
            step_fn, (s, weights, layer_start), np.arange(n_spikes)
        )
        return state, spikes

    def custom_trajectory_fwd(
        s: StepState,
        weights: List[WeightRecurrent],
        layer_start: int,
        known_spikes: Spike,
    ):
        # here the hardware call can be injected instead of calling forward
        output_state, spikes = custom_trajectory(s, weights, layer_start, known_spikes)
        return (output_state, spikes), (spikes, output_state, known_spikes)

    def custom_trajectory_bwd(res, g):
        spikes, (state, weights, layer_start), known_spikes = res
        (adjoint_state, grads, _), adjoint_spikes = g

        (adjoint_state, grads, layer_start), _ = jax.lax.scan(
            partial(step_bwd_wrapper, weights),
            (adjoint_state, grads, layer_start),
            (spikes, adjoint_spikes),
            reverse=True,
        )
        return adjoint_state, grads, 0, known_spikes

    custom_trajectory = jax.custom_vjp(custom_trajectory)
    custom_trajectory.defvjp(custom_trajectory_fwd, custom_trajectory_bwd)

    def apply_fn(
        layer_start: int, weights, input_spikes: EventPropSpike, known_spikes: Spike
    ):
        s = StepState(
            neuron_state=initial_state,
            time=0.0,
            input_queue=InputQueue(input_spikes),
        )
        _, spikes = custom_trajectory(s, weights, layer_start, known_spikes)
        return spikes

    return init_fn, apply_fn
