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

import math
from functools import partial
from typing import List, Optional

import jax
import jax.numpy as jnp
import numpy as np
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
from jaxsnn.event.flow import lif_exponential_flow, lif_exponential_flow_vec
from jaxsnn.event.stepping import step, step_existing
from jaxsnn.event.trajectory import trajectory
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
        t_max (float): Maximum simulation time
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
    solver = partial(ttfs_solver, params.tau_mem, params.tau_syn,
                     params.v_th)
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
    solver = partial(ttfs_solver, params.tau_mem, params.tau_syn,
                     params.v_th)
    batched_solver = partial(next_event, jax.vmap(solver, in_axes=(0, None)))
    transition = partial(transition_with_recurrence, params)
    step_fn = partial(step, dynamics, transition, t_max, batched_solver)

    hidden_size = jnp.sum(jnp.array(layers))
    apply_fn = trajectory(step_fn, hidden_size, n_spikes)
    init_fn = construct_recurrent_init_fn(layers, mean, std, duplication)

    return init_fn, apply_fn


def MultiPopulationRecurrentLIF(  # pylint: disable=too-many-arguments,too-many-locals
    layers: List[int],
    n_spikes: int,
    t_max: float,
    params_per_population: List[LIFParameters],
    mean: List[float],
    std: List[float],
    duplication: Optional[int] = None,
) -> SingleInitApply:
    """Recurrent LIF graph with one `LIFParameters` per population.

    Like `RecurrentLIF`, but each population in `layers` has its own
    `LIFParameters`. Per-population `tau_mem`, `tau_syn`, `v_th`, `v_leak`,
    and `v_reset` are concatenated into per-neuron arrays and threaded
    through a vec'd dynamics flow (`lif_exponential_flow_vec`) plus a
    per-neuron-vmapped TTFS solver.

    Constraints (v1):
      - Each population must satisfy `tau_mem == tau_syn` or
        `tau_mem == 2 * tau_syn` exactly (within 1e-6 relative tolerance).
        The analytical TTFS solver in `event.root.ttfs` only handles those
        two ratios; arbitrary ratios return `t_max` (no spike). Closing
        this constraint requires a Newton-based solver (planned as a
        separate follow-up).
      - Each population is internally homogeneous (one params per population,
        not per-neuron). Per-neuron heterogeneity within a population is also
        deferred to the same follow-up.
      - `v_th` must be 1.0 for every population. jaxsnn's `step` hard-codes
        a `V >= 1.0` threshold check that the TTFS solver's `v_th` cannot
        override; off-1.0 values silently desynchronise spike detection.

    Args:
        layers: Number of neurons per population.
        n_spikes: Maximum number of spikes simulated in the trajectory.
        t_max: Maximum simulation time.
        params_per_population: One `LIFParameters` per entry in `layers`.
        mean: Per-population weight init mean (forwarded to
            `construct_recurrent_init_fn`).
        std: Per-population weight init std (forwarded to
            `construct_recurrent_init_fn`).
        duplication: Optional input-weight duplication factor.

    Returns:
        SingleInitApply: Pair of init/apply functions, callable like
        `RecurrentLIF`.

    Raises:
        ValueError: If `layers` is empty or contains non-positive sizes,
            if `params_per_population` length does not match `layers`, if
            any population has non-positive `tau_mem` / `tau_syn`, if any
            population's `tau_mem / tau_syn` ratio is not 1 or 2 (within
            1e-6 relative tolerance), or if any population's `v_th` is
            not 1.0.
    """
    if not layers:
        raise ValueError("layers must be non-empty.")
    if any(not isinstance(n, (int, np.integer)) or n <= 0 for n in layers):
        raise ValueError(
            f"Each entry in layers must be a positive integer; got {list(layers)}."
        )
    if len(params_per_population) != len(layers):
        raise ValueError(
            f"params_per_population length ({len(params_per_population)}) "
            f"must match layers length ({len(layers)})."
        )

    for i, p in enumerate(params_per_population):
        tm = float(p.tau_mem)
        ts = float(p.tau_syn)
        if tm <= 0 or ts <= 0:
            raise ValueError(
                f"Population {i} has non-positive time constant(s): "
                f"tau_mem={p.tau_mem}, tau_syn={p.tau_syn}. Both must be > 0."
            )
        r = tm / ts
        if not (math.isclose(r, 1.0, rel_tol=1e-6)
                or math.isclose(r, 2.0, rel_tol=1e-6)):
            raise ValueError(
                f"Population {i} has tau_mem/tau_syn ratio {r} "
                f"(tau_mem={p.tau_mem}, tau_syn={p.tau_syn}); the analytical "
                "TTFS solver only handles ratios exactly 1 or 2 (within 1e-6 "
                "relative tolerance). Use tau_mem == tau_syn or tau_mem == 2 * "
                "tau_syn per population, or wait for the Newton-solver follow-up."
            )
        # v_th must be exactly 1.0: jaxsnn's `step.py` hard-codes `V >= 1.`
        # for spike detection (see jaxsnn.event.stepping.step:85). The TTFS
        # solver is configurable via params.v_th, but the trajectory layer
        # is not. Off-1.0 v_th silently desynchronises the two layers.
        if not math.isclose(float(p.v_th), 1.0, rel_tol=1e-6):
            raise ValueError(
                f"Population {i} has v_th={p.v_th}; only v_th=1.0 is "
                "supported (jaxsnn's `step` hard-codes a V>=1.0 threshold "
                "check that the TTFS solver's v_th cannot override). "
                "Off-1.0 values produce inconsistent spike detection."
            )

    # Concatenate per-population fields into per-neuron arrays.
    def _expand(field):
        chunks = [
            jnp.full((n,), float(getattr(p, field)))
            for n, p in zip(layers, params_per_population)
        ]
        return jnp.concatenate(chunks)

    per_neuron_params = LIFParameters(
        tau_syn=_expand("tau_syn"),
        tau_mem=_expand("tau_mem"),
        v_th=_expand("v_th"),
        v_leak=_expand("v_leak"),
        v_reset=_expand("v_reset"),
    )

    # Vec'd dynamics: per-neuron kernel built once, expm applied per neuron.
    dynamics = lif_exponential_flow_vec(per_neuron_params)

    # Per-neuron solver: vmap ttfs_solver over (tau_mem, tau_syn, v_th, state).
    def _per_neuron_solver(tau_mem_n, tau_syn_n, v_th_n, state_n, t_max_):
        return ttfs_solver(tau_mem_n, tau_syn_n, v_th_n, state_n, t_max_)

    _vec_solver = jax.vmap(_per_neuron_solver, in_axes=(0, 0, 0, 0, None))

    def step_solver(state, t_max_):
        return _vec_solver(
            per_neuron_params.tau_mem,
            per_neuron_params.tau_syn,
            per_neuron_params.v_th,
            state,
            t_max_,
        )

    batched_solver = partial(next_event, step_solver)

    # Transition uses per-neuron `v_reset` via broadcasting in `jnp.where`.
    transition = partial(transition_with_recurrence, per_neuron_params)
    step_fn = partial(step, dynamics, transition, t_max, batched_solver)

    hidden_size = int(sum(layers))
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
    solver = partial(ttfs_solver, params.tau_mem, params.tau_syn,
                     params.v_th)
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
    solver = partial(ttfs_solver, params.tau_mem, params.tau_syn,
                     params.v_th)
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

    size = jnp.sum(jnp.array(layers))

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

    size = jnp.sum(jnp.array(layers))

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
