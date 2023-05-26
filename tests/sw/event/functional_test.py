import jax.numpy as np
import jax
from functools import partial
from jaxsnn.event.functional import exponential_flow, step
from jaxsnn.event.root.ttfs import ttfs_solver
from jaxsnn.event.leaky_integrate_and_fire import transition_without_recurrence
from jaxsnn.functional.leaky_integrate_and_fire import LIFState, LIFParameters
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from jaxsnn.base.types import StepState, Spike, InputQueue, WeightInput


def test_exponential_flow():
    A = np.array([[-1, 1], [0, -1]])
    flow_function = exponential_flow(A)
    state = LIFState(V=1.0, I=1.0)
    new_state = flow_function(state, 1.0)
    assert_almost_equal(new_state.I, 0.368, 3)
    assert_almost_equal(new_state.V, 0.736, 3)


def test_batched_exponential_flow():
    A = np.array([[-1, 1], [0, -1]])
    flow_function = exponential_flow(A)
    state = LIFState(V=np.full(10, 1.0), I=np.full(10, 1.0))
    new_state = flow_function(state, 1.0)
    assert_array_almost_equal(new_state.I, np.full(10, 0.368), 3)
    assert_array_almost_equal(new_state.V, np.full(10, 0.736), 3)


def test_step():
    p = LIFParameters()
    A = np.array([[-p.tau_mem_inv, p.tau_mem_inv], [0, -p.tau_syn_inv]])
    flow = exponential_flow(A)
    dynamics = jax.vmap(flow, in_axes=(0, None))

    t_max = 10.0
    n_input = 2
    n_hidden = 2
    start_time = 0.0
    layer_start = 2

    solver = partial(ttfs_solver, p.tau_mem, p.v_th)
    batched_solver = jax.vmap(solver, in_axes=(0, None))
    transition = partial(transition_without_recurrence, p)
    step_fn = partial(step, dynamics, transition, t_max, batched_solver)
    weights = WeightInput(np.zeros((n_input, n_hidden)))
    neuron_state = LIFState(np.zeros(n_hidden), np.zeros(n_hidden))

    spikes = Spike(time=np.array([1.0, 2.0]), idx=np.array([0, 1]))
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


def test_step_same_time():
    p = LIFParameters()
    A = np.array([[-p.tau_mem_inv, p.tau_mem_inv], [0, -p.tau_syn_inv]])
    flow = exponential_flow(A)
    dynamics = jax.vmap(flow, in_axes=(0, None))

    t_max = 10.0
    n_input = 2
    n_hidden = 2
    start_time = 0.0
    layer_start = 2

    solver = partial(ttfs_solver, p.tau_mem, p.v_th)
    batched_solver = jax.vmap(solver, in_axes=(0, None))
    transition = partial(transition_without_recurrence, p)
    step_fn = partial(step, dynamics, transition, t_max, batched_solver)
    weights = WeightInput(np.zeros((n_input, n_hidden)))
    neuron_state = LIFState(np.zeros(n_hidden), np.zeros(n_hidden))

    spikes = Spike(time=np.array([1.0, 2.0]), idx=np.array([0, 1]))
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
    p = LIFParameters()
    A = np.array([[-p.tau_mem_inv, p.tau_mem_inv], [0, -p.tau_syn_inv]])
    flow = exponential_flow(A)
    dynamics = jax.vmap(flow, in_axes=(0, None))

    t_max = 10.0
    n_input = 2
    n_hidden = 2
    start_time = 0.0
    layer_start = 2

    solver = partial(ttfs_solver, p.tau_mem, p.v_th)
    batched_solver = jax.vmap(solver, in_axes=(0, None))
    transition = partial(transition_without_recurrence, p)
    step_fn = partial(step, dynamics, transition, t_max, batched_solver)
    weights = WeightInput(np.ones((n_input, n_hidden)))
    neuron_state = LIFState(np.zeros(n_hidden), np.zeros(n_hidden))

    # normal input spike, neuron current should increase
    spikes = Spike(time=np.array([1.0]), idx=np.array([0]))
    state = StepState(neuron_state, start_time, InputQueue(spikes))
    step_state = (state, weights, layer_start)

    (step_state, weights, _), spike = step_fn(step_state)
    assert spike.time == 1.0
    assert spike.idx == 0
    assert step_state.input_queue.head == 1
    assert np.all(step_state.neuron_state.I == np.ones(2))

    # input spike of previous layer, neuron current should not increase
    spikes = Spike(time=np.array([1.0]), idx=np.array([0]))
    state = StepState(neuron_state, start_time, InputQueue(spikes))
    layer_start = 5
    step_state = (state, weights, layer_start)

    (step_state, weights, _), spike = step_fn(step_state)
    assert spike.time == 1.0
    assert spike.idx == 0
    assert step_state.input_queue.head == 1
    assert np.all(step_state.neuron_state.I == np.zeros(2))
