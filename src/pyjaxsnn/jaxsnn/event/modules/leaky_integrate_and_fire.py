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
from typing import List, Optional

import jax
import jax.numpy as np
from jaxsnn.base.params import LIFParameters
from jaxsnn.event.adjoint_lif import (
    adjoint_lif_exponential_flow,
    adjoint_transition_with_recurrence,
    adjoint_transition_without_recurrence,
    step_bwd,
    construct_adjoint_apply_fn
)
from jaxsnn.event.construct import (
    construct_init_fn,
    construct_recurrent_init_fn,
)
from jaxsnn.event.flow import lif_exponential_flow
from jaxsnn.event.functional import step, step_existing, trajectory
from jaxsnn.event.root import ttfs_solver
from jaxsnn.event.root.next_finder import next_event
from jaxsnn.event.transition import (
    transition_with_recurrence,
    transition_without_recurrence,
)
from jaxsnn.event.types import (
    SingleInitApply,
    SingleInitApplyHW,
)


def LIF(  # pylint: disable=too-many-arguments
    size: int,
    n_spikes: int,
    t_max: float,
    params: LIFParameters,
    mean: float = 0.5,
    std: float = 2.0,
    duplication: Optional[int] = None,
) -> SingleInitApply:
    """A feed-forward layer of LIF Neurons.

    Args:
        size (int): Number of hidden neurons
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
    step_fn = partial(step, dynamics, transition, t_max, batched_solver)

    apply_fn = trajectory(step_fn, size, n_spikes)
    init_fn = construct_init_fn(size, mean, std, duplication)

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
    step_fn = partial(step, dynamics, transition, t_max, batched_solver)

    hidden_size = np.sum(np.array(layers))
    apply_fn = trajectory(step_fn, hidden_size, n_spikes)
    init_fn = construct_recurrent_init_fn(layers, mean, std, duplication)

    return init_fn, apply_fn


def EventPropLIF(  # pylint: disable=too-many-arguments,too-many-locals
    size: int,
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
        size (int): Number of hidden neurons
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
    step_fn = partial(step, dynamics, transition, t_max, batched_solver)

    # define adjoint step function
    single_adjoint_flow = adjoint_lif_exponential_flow(params)
    adjoint_dynamics = jax.vmap(single_adjoint_flow, in_axes=(0, None))
    adjoint_tr_dynamics = partial(
        adjoint_transition_without_recurrence, params
    )
    step_fn_bwd = partial(
        step_bwd, adjoint_dynamics, adjoint_tr_dynamics, t_max
    )

    init_fn = construct_init_fn(size, mean, std, duplication)
    apply_fn = construct_adjoint_apply_fn(
        step_fn, step_fn_bwd, size, n_spikes, wrap_only_step
    )

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
    step_fn = partial(step, dynamics, transition, t_max, batched_solver)

    # define adjoint step function
    single_adjoint_flow = adjoint_lif_exponential_flow(params)
    adjoint_dynamics = jax.vmap(single_adjoint_flow, in_axes=(0, None))
    adjoint_tr_dynamics = partial(adjoint_transition_with_recurrence, params)
    step_fn_bwd = partial(
        step_bwd, adjoint_dynamics, adjoint_tr_dynamics, t_max
    )

    size = np.sum(np.array(layers))

    init_fn = construct_recurrent_init_fn(layers, mean, std, duplication)
    apply_fn = construct_adjoint_apply_fn(
        step_fn, step_fn_bwd, size, n_spikes, wrap_only_step
    )

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
    step_fn = partial(step_existing, dynamics, transition, t_max, None)

    single_adjoint_flow = adjoint_lif_exponential_flow(params)
    adjoint_dynamics = jax.vmap(single_adjoint_flow, in_axes=(0, None))
    adjoint_tr_dynamics = partial(adjoint_transition_with_recurrence, params)

    step_fn_bwd = partial(
        step_bwd, adjoint_dynamics, adjoint_tr_dynamics, t_max
    )

    size = np.sum(np.array(layers))

    init_fn = construct_recurrent_init_fn(layers, mean, std, duplication)
    apply_fn = construct_adjoint_apply_fn(
        step_fn, step_fn_bwd, size, n_spikes
    )
    return init_fn, apply_fn


def HardwareLIF(  # pylint: disable=too-many-arguments,too-many-locals
    size: int,
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
    step_fn = partial(step_existing, dynamics, transition, t_max, None)

    # define adjoint step function (EventProp)
    single_adjoint_flow = adjoint_lif_exponential_flow(params)
    adjoint_dynamics = jax.vmap(single_adjoint_flow, in_axes=(0, None))
    adjoint_tr_dynamics = partial(
        adjoint_transition_without_recurrence, params
    )

    step_fn_bwd = partial(
        step_bwd, adjoint_dynamics, adjoint_tr_dynamics, t_max
    )

    init_fn = construct_init_fn(size, mean, std, duplication)
    apply_fn = construct_adjoint_apply_fn(
        step_fn, step_fn_bwd, size, n_spikes
    )

    return init_fn, apply_fn
