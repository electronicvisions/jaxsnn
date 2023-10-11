from functools import partial

import jax
import jax.numpy as np
from jaxsnn.base.params import LIFParameters
from jaxsnn.event.flow import exponential_flow
from jaxsnn.event.functional import step
from jaxsnn.event.leaky_integrate_and_fire import (
    LIFState,
    transition_without_recurrence,
)
from jaxsnn.event.root.next import next_event
from jaxsnn.event.root.ttfs import ttfs_solver
from jaxsnn.event.types import (
    EventPropSpike,
    InputQueue,
    StepState,
    WeightInput,
)


def test_step():
    params = LIFParameters()
    kernel = np.array(
        [[-params.tau_mem_inv, params.tau_mem_inv], [0, -params.tau_syn_inv]]
    )
    flow = exponential_flow(kernel)
    dynamics = jax.vmap(flow, in_axes=(0, None))

    t_max = 10.0
    n_input = 2
    n_hidden = 2
    start_time = 0.0
    layer_start = 2

    solver = partial(ttfs_solver, params.tau_mem, params.v_th)
    batched_solver = partial(next_event, jax.vmap(solver, in_axes=(0, None)))

    transition = partial(transition_without_recurrence, params)
    step_fn = partial(step, dynamics, transition, batched_solver, t_max)
    weights = WeightInput(np.zeros((n_input, n_hidden)))
    neuron_state = LIFState(np.zeros(n_hidden), np.zeros(n_hidden))

    spikes = EventPropSpike(
        time=np.array([1.0, 2.0]),
        idx=np.array([0, 1]),
        current=np.array([0.0, 0.0]),
    )
    state = StepState(neuron_state, start_time, InputQueue(spikes))
    step_state = (state, weights, layer_start)

    step_state, spike = step_fn(step_state)
    assert spike.time == 1.0
    assert spike.idx == 0

    step_state, spike = step_fn(step_state)
    assert spike.time == 2.0
    assert spike.idx == 1

    step_state, spike = step_fn(step_state)
    assert spike.time == t_max
    assert spike.idx == -1


def test_step_no_transition():
    params = LIFParameters()
    kernel = np.array(
        [[-params.tau_mem_inv, params.tau_mem_inv], [0, -params.tau_syn_inv]]
    )
    flow = exponential_flow(kernel)
    dynamics = jax.vmap(flow, in_axes=(0, None))

    t_max = 10.0
    n_input = 2
    n_hidden = 2
    start_time = 0.0
    layer_start = 2

    solver = partial(ttfs_solver, params.tau_mem, params.v_th)
    batched_solver = partial(next_event, jax.vmap(solver, in_axes=(0, None)))

    transition = partial(transition_without_recurrence, params)
    step_fn = partial(step, dynamics, transition, batched_solver, t_max)
    weights = WeightInput(np.ones((n_input, n_hidden)))
    neuron_state = LIFState(np.zeros(n_hidden), np.zeros(n_hidden))

    # normal input spike, neuron current should increase
    spikes = EventPropSpike(
        time=np.array([1.0]), idx=np.array([0]), current=np.array([0.0])
    )
    state = StepState(neuron_state, start_time, InputQueue(spikes))
    step_state = (state, weights, layer_start)

    (step_state, weights, _), spike = step_fn(step_state)
    assert spike.time == 1.0
    assert spike.idx == 0
    assert step_state.input_queue.head == 1
    assert np.all(step_state.neuron_state.I == np.ones(2))

    # input spike of previous layer, neuron current should not increase
    spikes = EventPropSpike(
        time=np.array([1.0]), idx=np.array([0]), current=np.array([0.0])
    )
    state = StepState(neuron_state, start_time, InputQueue(spikes))
    layer_start = 5
    step_state = (state, weights, layer_start)

    (step_state, weights, _), spike = step_fn(step_state)
    assert spike.time == 1.0
    assert spike.idx == 0
    assert step_state.input_queue.head == 1
    assert np.all(step_state.neuron_state.I == np.zeros(2))
