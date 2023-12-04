# pylint: disable=invalid-name
"""Implement different LIF layers, which can be concatenated

Each layer returns a paif or two functions, the `init` function and the
`apply` function. These functions can be concatenated with
`jaxsnn.event.compose.serial`, which also returns and init/apply pair,
consisting of multiple layers. The `init` function is used to initalize the
weights of the network. The `apply` function does the inference and is
equivalent to the forward function is in PyTorch. It receives the input
spikes and weights of the network and returns the hidden spikes.

The layers in this module differ in the topology they offer (feed-forward /
recurrent) and in the way the gradients are computed (analytical via jax.grad
or with an adjoint system (EventProp).

`HardwareLIF` and `HardwareRecurrentLIF` allow the execution of the forward
pass on the neuromorphic BSS-2 system. They forward pass is executed on the
neuromorphic system and the spikes are retrived. Because the spike data from
BSS-2 is missing information about the synaptic current at spike time (which
is needed for the EventProp algorithm), a second forward pass in software is
executed. The spike times from the hardware are used as solution for the root
solving. The adjoint system of the EventProp algorithm is added as a custom
Vector-Jacobian-Product (VJP).
"""

from functools import partial
from typing import List, Optional, Tuple

import jax
import jax.numpy as np
from jaxsnn.base.params import LIFParameters
from jaxsnn.event.adjoint_lif import (
    adjoint_lif_exponential_flow,
    adjoint_transition_with_recurrence,
    adjoint_transition_without_recurrence,
    step_bwd,
)
from jaxsnn.event.construct import (
    construct_init_fn,
    construct_recurrent_init_fn,
)
from jaxsnn.event.flow import lif_exponential_flow
from jaxsnn.event.functional import StepInput, step, trajectory
from jaxsnn.event.root import ttfs_solver
from jaxsnn.event.root.next import next_event, next_queue
from jaxsnn.event.transition import (
    transition_with_recurrence,
    transition_without_recurrence,
)
from jaxsnn.event.types import (
    EventPropSpike,
    InputQueue,
    LIFState,
    SingleInitApply,
    SingleInitApplyHW,
    Spike,
    StepState,
    Weight,
)


def LIF(  # pylint: disable=too-many-arguments
    n_hidden: int,
    n_spikes: int,
    t_max: float,
    params: LIFParameters,
    mean: float = 0.5,
    std: float = 2.0,
    duplication: Optional[int] = None,
) -> SingleInitApply:
    """A feed-forward layer of LIF Neurons.

    Args:
        n_hidden (int): Number of hidden neurons
        n_spikes (int): Number of spikes which are simulated in this layer
        t_max (float): Maxium simulation time
        p (LIFParameters): Parameters of the LIF neurons
        mean (float, optional): Mean of initial weights. Defaults to 0.5.
        std (float, optional): Standard deviation of initial weights.
            Defaults to 2.0.

    Returns:
        SingleInitApply: _description_
    """
    single_flow = lif_exponential_flow(params)
    dynamics = jax.vmap(single_flow, in_axes=(0, None))

    # construct step function
    solver = partial(ttfs_solver, params.tau_mem, params.v_th)
    batched_solver = partial(next_event, jax.vmap(solver, in_axes=(0, None)))
    transition = partial(transition_without_recurrence, params)
    step_fn = partial(step, dynamics, transition, batched_solver, t_max)

    apply_fn = trajectory(step_fn, n_hidden, n_spikes)
    init_fn = construct_init_fn(n_hidden, mean, std, duplication)

    return init_fn, apply_fn


def RecurrentLIF(  # pylint: disable=too-many-arguments,too-many-locals
    layers: List[int],
    n_spikes: int,
    t_max: float,
    params: LIFParameters,
    mean: List[float],
    std: List[float],
    duplication: Optional[int] = None,
) -> SingleInitApply:
    single_flow = lif_exponential_flow(params)
    dynamics = jax.vmap(single_flow, in_axes=(0, None))

    # construct step function
    solver = partial(ttfs_solver, params.tau_mem, params.v_th)
    batched_solver = partial(next_event, jax.vmap(solver, in_axes=(0, None)))
    transition = partial(transition_with_recurrence, params)
    step_fn = partial(step, dynamics, transition, batched_solver, t_max)

    hidden_size = np.sum(np.array(layers))
    apply_fn = trajectory(step_fn, hidden_size, n_spikes)
    init_fn = construct_recurrent_init_fn(layers, mean, std, duplication)

    return init_fn, apply_fn


def EventPropLIF(  # pylint: disable=too-many-arguments,too-many-locals
    n_hidden: int,
    n_spikes: int,
    t_max: float,
    params: LIFParameters,
    mean=0.5,
    std=2.0,
    wrap_only_step: bool = False,
    duplication: Optional[int] = None,
) -> SingleInitApply:
    """Feed-forward layer of LIF neurons with EventProp gradient computation.

    Args:
        n_hidden (int): Number of hidden neurons
        n_spikes (int): Number of spikes which are simulated in this
        t_max (float): Maximum simulation time
        p (LIFParameters): Parameters of the LIF neurons
        mean (float, optional): Mean of initial weights. Defaults to 0.5.
        std (float, optional): Standard deviation of initial weights.
            Defaults to 2.0.
        wrap_only_step (bool, optional): If custom vjp should be defined
            only for the step function or for the entire trajectory.
            Defaults to False.
        duplication (Optional[int], optional): Factor with which input weights
            are duplicated. Defaults to None.

    Returns:
        SingleInitApply: Pair of init apply functions.
    """
    single_flow = lif_exponential_flow(params)
    dynamics = jax.vmap(single_flow, in_axes=(0, None))

    # define step function
    solver = partial(ttfs_solver, params.tau_mem, params.v_th)
    batched_solver = partial(next_event, jax.vmap(solver, in_axes=(0, None)))
    transition = partial(transition_without_recurrence, params)
    step_fn = partial(step, dynamics, transition, batched_solver, t_max)

    # define adjoint step function
    single_adjoint_flow = adjoint_lif_exponential_flow(params)
    adjoint_dynamics = jax.vmap(single_adjoint_flow, in_axes=(0, None))
    adjoint_tr_dynamics = partial(
        adjoint_transition_without_recurrence, params
    )
    step_fn_bwd = partial(
        step_bwd, adjoint_dynamics, adjoint_tr_dynamics, t_max
    )

    init_fn = construct_init_fn(n_hidden, mean, std, duplication)
    wrap_only_step = False

    # TODO
    # for defining a cusotm backward, one can either define the custom vjp
    # only for the step function, or define the custom vjp for the whole
    # trajectory / scan (like done below).
    # Only doing it for the step function shoud be equivalent but throws NaNs
    if wrap_only_step:
        # define custom forward to save data of forward pass
        def step_fn_fwd(*args):
            res = step_fn(*args)
            return res, res

        # define custom vjp
        step_fn_event_prop = jax.custom_vjp(step_fn)
        step_fn_event_prop.defvjp(step_fn_fwd, step_fn_bwd)

        forward = trajectory(step_fn_event_prop, n_hidden, n_spikes)
        return init_fn, forward

    # wrap step bwd so it is compliant with scan syntax
    def step_bwd_wrapper(
        weights: Weight,
        init: StepInput,
        xs: Tuple[EventPropSpike, EventPropSpike],
    ):
        adjoint_state, grads, layer_start = init
        spike, adjoint_spike = xs
        res = (None, weights, layer_start), spike
        g = (adjoint_state, grads, 0), adjoint_spike
        return step_fn_bwd(res, g)

    def custom_trajectory(s: StepState, weights: Weight, layer_start: int):
        return jax.lax.scan(
            step_fn, (s, weights, layer_start), np.arange(n_spikes)
        )

    def custom_trajectory_fwd(*args):
        # save res for backward
        res = custom_trajectory(*args)
        return res, res

    def custom_trajectory_bwd(res, g):
        (_, weights, layer_start), spikes = res
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

    def apply_fn(
        layer_start: int, weights: Weight, input_spikes: EventPropSpike
    ):
        initial_state = LIFState(np.zeros(n_hidden), np.zeros(n_hidden))
        s = StepState(
            neuron_state=initial_state,
            time=0.0,
            input_queue=InputQueue(input_spikes),
        )
        _, spikes = custom_trajectory(s, weights, layer_start)
        return spikes

    return init_fn, apply_fn


def RecurrentEventPropLIF(  # pylint: disable=too-many-arguments,too-many-locals
    layers: List[int],
    n_spikes: int,
    t_max: float,
    params: LIFParameters,
    mean: List[float],
    std: List[float],
    wrap_only_step: bool = False,
    duplication: Optional[int] = None,
) -> SingleInitApply:
    """Use quadrants of the recurrent weight matrix to set up a multi-layer
    feed-forward LIF in one recurrent layer.

    When simulating multiple layers, the first layer needs to be fully
    simulated before the resulting spikes are passed to the next layer. When
    viewing multiple feed-forward layers as one recurrent layer with the only
    rectangular parts of the weight matrix initialized with non-zero entries,
    multiple feed-forward layers can be simulated together.

    Args:
        layers (List[int]): Number of neurons in each feed-forward layer
        n_spikes (int): Number of spikes which are simulated in this
        t_max (float): Maximum simulation time
        p (LIFParameters): Parameters of the LIF neurons
        mean (float): Mean of initial weights.
        std (float): Standard deviation of initial weights.
        wrap_only_step (bool, optional): If custom vjp should be defined only
            for the step function or for the entire trajectory. Defaults
            to False.
        duplication (Optional[int], optional): Factor with which input weights
            are duplicated. Defaults to None.

    Returns:
        SingleInitApply: Pair of init apply functions.
    """
    single_flow = lif_exponential_flow(params)
    dynamics = jax.vmap(single_flow, in_axes=(0, None))

    # define step function
    solver = partial(ttfs_solver, params.tau_mem, params.v_th)
    batched_solver = partial(next_event, jax.vmap(solver, in_axes=(0, None)))
    transition = partial(transition_with_recurrence, params)
    step_fn = partial(step, dynamics, transition, batched_solver, t_max)

    # define adjoint step function
    single_adjoint_flow = adjoint_lif_exponential_flow(params)
    adjoint_dynamics = jax.vmap(single_adjoint_flow, in_axes=(0, None))
    adjoint_tr_dynamics = partial(adjoint_transition_with_recurrence, params)
    step_fn_bwd = partial(
        step_bwd, adjoint_dynamics, adjoint_tr_dynamics, t_max
    )

    hidden_size = np.sum(np.array(layers))
    initial_state = LIFState(np.zeros(hidden_size), np.zeros(hidden_size))

    init_fn = construct_recurrent_init_fn(layers, mean, std, duplication)
    wrap_only_step = False

    # TODO
    # for defining a custom backward, one can either define the custom vjp
    # only for the step function, or define the custom vjp for the whole
    # trajectory / scan (like done below).
    # Only doing it for the step function shoud be equivalent but throws NaNs
    if wrap_only_step:

        def step_fn_fwd(*args):
            res = step_fn(*args)
            return res, res

        # define custom vjp **only** for the step function
        step_fn_event_prop = jax.custom_vjp(step_fn)
        step_fn_event_prop.defvjp(step_fn_fwd, step_fn_bwd)

        forward = trajectory(step_fn_event_prop, hidden_size, n_spikes)
        return init_fn, forward

    # wrap step bwd so it is compliant with scan syntax
    def step_bwd_wrapper(
        weights: Weight,
        init: StepInput,
        xs: Tuple[EventPropSpike, EventPropSpike],
    ):
        adjoint_state, grads, layer_start = init
        spike, adjoint_spike = xs
        res = (None, weights, layer_start), spike
        g = (adjoint_state, grads, 0), adjoint_spike
        return step_fn_bwd(res, g)

    def custom_trajectory(s: StepState, weights: Weight, layer_start: int):
        return jax.lax.scan(
            step_fn, (s, weights, layer_start), np.arange(n_spikes)
        )

    def custom_trajectory_fwd(*args):
        res = custom_trajectory(*args)
        return res, res

    def custom_trajectory_bwd(res, g):
        (_, weights, layer_start), spikes = res
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

    def apply_fn(
        layer_start: int, weights: Weight, input_spikes: EventPropSpike
    ):
        s = StepState(
            neuron_state=initial_state,
            time=0.0,
            input_queue=InputQueue(input_spikes),
        )
        _, spikes = custom_trajectory(s, weights, layer_start)
        return spikes

    return init_fn, apply_fn


def HardwareRecurrentLIF(  # pylint: disable=too-many-arguments,too-many-locals
    layers: List[int],
    n_spikes: int,
    t_max: float,
    params: LIFParameters,
    mean: List[float],
    std: List[float],
    duplication: Optional[int] = None,
):
    single_flow = lif_exponential_flow(params)
    dynamics = jax.vmap(single_flow, in_axes=(0, None))
    transition = partial(transition_with_recurrence, params)

    single_adjoint_flow = adjoint_lif_exponential_flow(params)
    adjoint_dynamics = jax.vmap(single_adjoint_flow, in_axes=(0, None))
    adjoint_tr_dynamics = partial(adjoint_transition_with_recurrence, params)

    step_fn_bwd = partial(
        step_bwd, adjoint_dynamics, adjoint_tr_dynamics, t_max
    )

    hidden_size = np.sum(np.array(layers))
    initial_state = LIFState(np.zeros(hidden_size), np.zeros(hidden_size))

    init_fn = construct_recurrent_init_fn(layers, mean, std, duplication)

    # wrap step bwd so it is compliant with scan syntax
    def step_bwd_wrapper(weights, init, xs):
        adjoint_state, grads, layer_start = init
        spike, adjoint_spike = xs
        res = (None, weights, layer_start), spike
        g = (adjoint_state, grads, 0), adjoint_spike
        return step_fn_bwd(res, g)

    def custom_trajectory(
        s: StepState,
        weights: Weight,
        layer_start: int,
        known_spikes: Spike,
    ):
        # attach known spikes to "root solver"
        solver = partial(next_queue, known_spikes, layer_start)
        step_fn = partial(step, dynamics, transition, solver, t_max)
        state, spikes = jax.lax.scan(
            step_fn, (s, weights, layer_start), np.arange(n_spikes)
        )
        return state, spikes

    def custom_trajectory_fwd(
        s: StepState,
        weights: Weight,
        layer_start: int,
        known_spikes: Spike,
    ):
        # TODO, do we need to save the known spike for bwd?
        output_state, spikes = custom_trajectory(
            s, weights, layer_start, known_spikes
        )
        return (output_state, spikes), (output_state, spikes, known_spikes)

    def custom_trajectory_bwd(res, g):
        (_, weights, layer_start), spikes, known_spikes = res
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
        layer_start: int,
        weights: Weight,
        input_spikes: EventPropSpike,
        known_spikes: Spike,
    ):
        s = StepState(
            neuron_state=initial_state,
            time=0.0,
            input_queue=InputQueue(input_spikes),
        )
        _, spikes = custom_trajectory(s, weights, layer_start, known_spikes)
        return spikes

    return init_fn, apply_fn


def HardwareLIF(  # pylint: disable=too-many-arguments,too-many-locals
    n_hidden: int,
    n_spikes: int,
    t_max: float,
    params: LIFParameters,
    mean: float,
    std: float,
    duplication: Optional[int] = None,
) -> SingleInitApplyHW:
    # define step function
    single_flow = lif_exponential_flow(params)
    dynamics = jax.vmap(single_flow, in_axes=(0, None))
    transition = partial(transition_without_recurrence, params)

    # define adjoint step function (EventProp)
    single_adjoint_flow = adjoint_lif_exponential_flow(params)
    adjoint_dynamics = jax.vmap(single_adjoint_flow, in_axes=(0, None))
    adjoint_tr_dynamics = partial(
        adjoint_transition_without_recurrence, params
    )

    step_fn_bwd = partial(
        step_bwd, adjoint_dynamics, adjoint_tr_dynamics, t_max
    )

    initial_state = LIFState(np.zeros(n_hidden), np.zeros(n_hidden))
    init_fn = construct_init_fn(n_hidden, mean, std, duplication)

    # wrap step bwd so it is compliant with scan syntax
    def step_bwd_wrapper(weights, init, xs):
        adjoint_state, grads, layer_start = init
        spike, adjoint_spike = xs
        res = (None, weights, layer_start), spike
        g = (adjoint_state, grads, 0), adjoint_spike
        return step_fn_bwd(res, g)

    def custom_trajectory(
        s: StepState,
        weights: Weight,
        layer_start: int,
        known_spikes: Spike,
    ):
        # attach known spikes to "root solver"
        solver = partial(next_queue, known_spikes, layer_start)
        step_fn = partial(step, dynamics, transition, solver, t_max)
        state, spikes = jax.lax.scan(
            step_fn, (s, weights, layer_start), np.arange(n_spikes)
        )
        return state, spikes

    def custom_trajectory_fwd(
        s: StepState,
        weights: Weight,
        layer_start: int,
        known_spikes: Spike,
    ):
        # TODO, do we need to save the known spike for bwd?
        output_state, spikes = custom_trajectory(
            s, weights, layer_start, known_spikes
        )
        return (output_state, spikes), (spikes, output_state, known_spikes)

    def custom_trajectory_bwd(res, g):
        spikes, (_, weights, layer_start), known_spikes = res
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
        layer_start: int,
        weights: Weight,
        input_spikes: EventPropSpike,
        known_spikes: Spike,
    ):
        s = StepState(
            neuron_state=initial_state,
            time=0.0,
            input_queue=InputQueue(input_spikes),
        )
        _, spikes = custom_trajectory(s, weights, layer_start, known_spikes)
        return spikes

    return init_fn, apply_fn
