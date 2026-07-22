from typing import List, Callable, Dict
from functools import partial

import jax
import jax.numpy as jnp

from jaxsnn.event.adjoint.lif.dynamics import adjoint_lif_exponential_flow
from jaxsnn.event.adjoint.lif.add_grads import add_grads
from jaxsnn.event.adjoint.step import adjoint_step
from jaxsnn.event.adjoint.lif.transition import adjoint_transition
from jaxsnn.event.solver.next_finder import next_event
from jaxsnn.event.stepping import (
    step,
    next_input,
    min_delay_check,
)
from jaxsnn.event.functional.lif.transition import transition
from jaxsnn.event.functional.lif.dynamics import lif_exponential_flow
from jaxsnn.event.modules.lif.parameters import LIFParameters
from jaxsnn.event.solver import ttfs_solver
from jaxsnn.event.states import LIFState
from jaxsnn.event.types import (
    Spike,
    StepState,
    Population,
    EventStepFn,
    AdjointStepFn,
)


def LIF(  # pylint: disable=invalid-name
    size: int,
    n_steps: int,
    params: LIFParameters,
) -> Population:
    """
    Creates a LIF layer for event-based simulation and backpropagation.
    Returns a generator function that builds the layer based on input
    connections, delays, and chosen backpropagation strategy. Supports
    both forward-only simulation and custom backward passes (eventprop).

    :param size: Number of neurons in the layer.
    :param n_steps: Number of event steps in the simulation.
    :param params: Parameters for the LIF neuron dynamics.

    :returns: A Population object containing the generator and parameters.
    """
    # pylint: disable=too-many-arguments
    def generator(
        pre_layer_pop_nodes: List[str],
        pre_layer_param_nodes: List[str],
        pre_layer_params: Dict[str, float],
        node_index_mapping: Dict[str, int],
        t_max: float,
        node: str,
        scc_mask: List[str],
        backprop_method: str
    ) -> Population.Functions:
        """
        Generate the initialization and apply functions for a neuron layer.
        Builds layer-specific step functions and forward/backward dynamics
        depending on the topology and backpropagation method.

        :param pre_layer_pop_nodes: List of names of input population nodes.
        :param pre_layer_param_nodes: List of names of input parameter nodes.
        :param pre_layer_params: Dictionary of synaptic parameters (delays).
        :param node_index_mapping: Mapping from node names to indices.
        :param t_max: Maximum simulation time.
        :param node: Name of this layer/node in the topology.
        :param scc_mask: List of nodes marking recurrent inputs for SCC
            handling.
        :param backprop_method: Backpropagation method, "analytical" or
            "eventprop".

        :returns: A Population.Functions object containing init, state, step,
            event, and adjoint_step functions.
        """
        def init_fn(
            rng: jax.Array,  # pylint: disable=unused-argument
        ) -> None:
            return None

        def state_fn() -> StepState:
            state = StepState(
                LIFState(jnp.zeros(size), jnp.zeros(size)),
                jnp.array(0.0)
            )
            return state

        def event_fn(
            n_steps: int,
        ) -> Spike:
            return Spike.empty(n_steps)

        # Initialize step function
        step_fn = build_step_functions(
            pre_layer_pop_nodes,
            pre_layer_param_nodes,
            pre_layer_params,
            scc_mask,
            params,
            t_max,
            node,
        )

        adjoint_step_fn = None
        if backprop_method == 'eventprop':
            # Build adjoint step function for eventprop
            adjoint_step_fn = build_adjoint_step_function(
                pre_layer_pop_nodes,
                pre_layer_param_nodes,
                node_index_mapping,
                params,
                t_max
            )

        return Population.Functions(
            init=init_fn,
            state=state_fn,
            step=step_fn,
            event=event_fn,
            adjoint_step=adjoint_step_fn,
        )

    parameters = {
        "size": size,
        "n_steps": n_steps,
        "lif_params": params,
    }

    return Population(generator, parameters, size, n_steps)


# pylint: disable=too-many-arguments, too-many-locals
def build_step_functions(
    pre_layer_pop_nodes: List[str],
    pre_layer_param_nodes: List[str],
    pre_layer_params: Dict[str, float],
    scc_mask: List[str],
    lif_params: LIFParameters,
    t_max: float,
    node: str,
) -> EventStepFn:
    """
    Construct the recurrent and non-recurrent step functions for a LIF layer.
    This function builds the two core step functions required to simulate a
    spiking neuron layer in both recurrent and feedforward cases. It assembles
    the solver for finding next events, sets up the neuron dynamics,
    builds the per-connection transition functions, and incorporates minimum
    delay checks as defined by the graph structure and neuron parameters.
    The recurrent step function is used when a layer is part of a strongly
    connected component (SCC), and the non-recurrent function wraps it for use
    in feedforward contexts.

    :param pre_layer_pop_nodes: List of names of input population nodes.
    :param pre_layer_param_nodes: List of names of input parameter nodes.
    :param pre_layer_params: Dictionary of synaptic parameters (delays).
    :param scc_mask: List of nodes indicating which inputs belong to SCC.
    :param lif_params: Parameters for the LIF neuron model.
    :param t_max: Maximum simulation time.
    :param node: Name of the current layer/node.

    :returns: The step function for the layer.
    """
    # Vectorized flow dynamics
    single_flow = lif_exponential_flow(
        lif_params.tau_syn,
        lif_params.tau_mem,
    )
    dynamics = jax.vmap(single_flow, in_axes=(0, None))

    # Solver for next event
    solver = partial(
        ttfs_solver,
        lif_params.tau_mem,
        lif_params.tau_syn,
        lif_params.v_th,
    )
    batched_solver = partial(
        next_event,
        jax.vmap(solver, in_axes=(0, None))
    )

    # Transitions per input connection
    transitions = [
        partial(transition, lif_params.v_reset, pre_layer_node)
        for pre_layer_node in pre_layer_param_nodes
    ]

    # Minimum delays
    min_delays = dict(pre_layer_params.items())

    # Next input function
    next_input_fn = partial(next_input, pre_layer_pop_nodes, min_delays)

    # Masked inputs and delays for min-delay check
    filtered_input_connections = [
        x for x in pre_layer_pop_nodes if x in scc_mask
    ]
    if (
        len(filtered_input_connections) == 1
        and filtered_input_connections[0] == node
    ):
        filtered_input_connections = []
    filtered_min_delays = {
        x: min_delays[x] for x in filtered_input_connections
    }
    min_delay_check_fn = partial(
        min_delay_check,
        filtered_input_connections,
        filtered_min_delays,
    )

    # Step functions
    step_fn = partial(
        step,
        next_input_fn,
        min_delay_check_fn,
        dynamics,
        transitions,
        t_max,
        batched_solver,
    )

    return step_fn


def build_apply_function(
    non_recurrent_step_fn: Callable,
    n_steps: int,
    layer_idx: int,
) -> Callable:
    """
    Construct the apply function for a LIF layer.
    Builds the forward pass function for a layer, scanning over time
    using the provided non-recurrent step function. The apply function
    handles integration over simulation time steps and manages the
    propagation of spikes, states, and queue indices.
    The result is a function that applies the spiking neuron dynamics
    to given weights, input spikes, and internal state.

    :param non_recurrent_step_fn: Step function to apply at each time step
        for the layer's dynamics.
    :param n_steps: Number of simulation time steps.
    :param layer_idx: Index of the current layer to apply the function to.

    :returns: A callable that executes the forward pass over `n_steps` for
        the specified layer.
    """
    def apply(weights, spikes, state, queue_heads):
        queue_indices = jnp.zeros(n_steps, dtype=int)
        carry = (
            weights,
            spikes,
            state,
            0,          # step index (not currently used inside step)
            layer_idx,
            queue_heads,
            queue_indices,
        )
        carry, _ = jax.lax.scan(
            non_recurrent_step_fn,
            carry,
            jnp.arange(n_steps),
        )
        spikes_out = carry[1]
        states_out = carry[2]
        queue_indices_out = carry[6]
        return spikes_out, states_out, weights, queue_indices_out

    return apply


def build_adjoint_step_function(
    pre_layer_pop_nodes: List[str],
    pre_layer_param_nodes: List[str],
    node_index_mapping: Dict[str, int],
    lif_params: LIFParameters,
    t_max: float,
) -> AdjointStepFn:
    """
    Build the adjoint step function used for backprop in spiking neurons.
    Constructs the adjoint step function by composing the adjoint neuron
    dynamics, adjoint transition dynamics, and gradient accumulation
    functions for each input connection. This function is used in the
    backward pass during event-based backpropagation.

    :param pre_layer_pop_nodes: List of names of input population nodes.
    :param pre_layer_param_nodes: List of names of input parameter nodes.
    :param node_index_mapping: Mapping from node names to indices.
    :param lif_params: Parameters of the LIF neuron model.
    :param t_max: Maximum simulation time.

    :returns: A callable adjoint step function for use in backward scans.
    """
    # Create adjoint neuron dynamics for vectorized application
    single_adjoint_flow = adjoint_lif_exponential_flow(
        lif_params.tau_syn,
        lif_params.tau_mem,
    )
    adjoint_dynamics = jax.vmap(single_adjoint_flow, in_axes=(0, None))

    # Create functions to accumulate gradients for each input connection
    add_grads_fns = [
        partial(add_grads, jnp.array(lif_params.tau_syn), pop_node, param_node)
        for pop_node, param_node in zip(
            pre_layer_pop_nodes, pre_layer_param_nodes)
    ]

    # Compose adjoint transition dynamics with gradient accumulation
    adjoint_tr_dynamics = partial(
        adjoint_transition,
        jnp.array(lif_params.v_th),
        jnp.array(lif_params.v_reset),
        jnp.array([node_index_mapping[node] for node in pre_layer_pop_nodes]),
        add_grads_fns
    )

    # Build the full adjoint step function for backward passes
    adjoint_step_fn = partial(
        adjoint_step,
        adjoint_dynamics,
        adjoint_tr_dynamics,
        t_max,
    )
    return adjoint_step_fn


def build_custom_vjp_apply_function(
    apply_fn: Callable,
    wrapped_adjoint_step_fn: Callable,
    n_steps: int,
) -> Callable:
    """
    Build a custom VJP-enabled apply function for the LIF layer.
    Wraps the forward apply function to support event-based backpropagation
    by defining a custom vector-Jacobian product (VJP). Uses the provided
    wrapped adjoint step function to scan backwards through time steps.

    :param apply_fn: The forward apply function (without VJP).
    :param wrapped_adjoint_step_fn: The adjoint step function wrapped with
        layer index, used for the backward scan.
    :param n_steps: Number of simulation time steps.

    :returns: A callable apply function with custom VJP defined.
    """
    def apply_fwd(weights, spikes, state, queue_heads):
        res = apply_fn(weights, spikes, state, queue_heads)
        return res, res

    def apply_bwd(res, grad):
        adjoint_spikes, adjoint_state, grads, _ = grad
        carry = (
            res[2],  # weights
            res[0],  # spikes
            res[3],  # queue indices
            adjoint_spikes,
            adjoint_state,
            grads,
        )
        carry, _ = jax.lax.scan(
            wrapped_adjoint_step_fn,
            carry,
            jnp.arange(n_steps - 1, -1, -1)
        )
        return carry[5], carry[3], carry[4], None

    apply_vjp = jax.custom_vjp(apply_fn)
    apply_vjp.defvjp(apply_fwd, apply_bwd)

    return apply_vjp
