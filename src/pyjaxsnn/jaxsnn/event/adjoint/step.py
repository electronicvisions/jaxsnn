from typing import Callable, Tuple, Dict, List

import jax
from jaxsnn.event.types import (
    EventState,
    StepState,
    Parameters,
    EventData,
    QueueHead,
    QueueIndex,
    IOData,
    States,
    AdjointStepFn,
)


def multi_layer_adjoint_step(  # pylint: disable=too-many-locals
    adjoint_step_fns: Dict[str, AdjointStepFn],
    nodes: List[str],
    carry: Tuple[Parameters, IOData, QueueIndex, IOData, States, Parameters],
    step_idx: int,
) -> Tuple[Tuple, int]:
    """
    Perform one adjoint backward step through all layers in reverse order.

    :param adjoint_step_fns: Dictionary of adjoint step functions per layer.
    :param nodes: List of node names (layers) to process.
    :param carry: Tuple containing weights, spikes, queue indices,
        adjoint spikes, adjoint states, and grads.
    :param step_idx: Index of the current step.
    :returns: Updated carry tuple and zero as dummy scan output.
    """

    (
        weights,
        spikes,
        queue_indices,
        adjoint_spikes,
        adjoint_states,
        grads
    ) = carry

    for i in range(len(nodes) - 1, -1, -1):
        node = nodes[i]
        layer_step = adjoint_step_fns[node]

        res = layer_step(
            (
                weights,
                spikes[node][step_idx],  # current spike
                queue_indices[node][step_idx],  # queue_index
                adjoint_spikes[node][step_idx],  # current adjoint_spike
                adjoint_states[node],  # adjoint_state
                grads,  # grads
                adjoint_spikes,  # adjoint_spikes for whole layer
            )
        )

        adjoint_spikes = res[1]
        adjoint_states[node] = res[2]
        grads = res[3]

    carry = (
        weights,
        spikes,
        queue_indices,
        adjoint_spikes,
        adjoint_states,
        grads,
    )

    return carry, 0


def adjoint_step(
    adjoint_dynamics: Callable[[EventState, jax.Array], EventState],
    adjoint_tr_dynamics: Callable[
        [
            StepState,
            EventData,
            EventData,
            QueueIndex,
            Parameters,
            Parameters,
            IOData,
        ],
        Tuple[StepState, Parameters, IOData]
    ],
    t_max: float,
    adjoint_step_input: Tuple[
        Parameters,
        EventData,
        QueueIndex,
        EventData,
        StepState,
        Parameters,
        EventData
    ],
) -> Tuple[Parameters, StepState, QueueHead, QueueIndex]:
    """
    Perform one backward adjoint step integrating dynamics and transitions.

    :param adjoint_dynamics: Function integrating adjoint neuron dynamics.
    :param adjoint_tr_dynamics: Function applying adjoint transition.
    :param t_max: Maximum simulation time.
    :param adjoint_step_input: Tuple containing
        (weights, spike, queue_index, adjoint_spike, adjoint_state, grads,
         adjoint_spikes).
    :returns: Tuple with updated
        (weights, adjoint_spikes, adjoint_state, grads).
    """
    (
        weights,
        spike,
        queue_index,
        adjoint_spike,
        adjoint_state,
        grads,
        adjoint_spikes,
    ) = adjoint_step_input

    reversed_time = t_max - spike.time

    time_diff = reversed_time - adjoint_state.time

    # integrate lambdas to the spike
    adjoint_state.neuron_state = adjoint_dynamics(
        adjoint_state.neuron_state, time_diff
    )
    adjoint_state.time = reversed_time

    no_event = spike.idx == -1

    tr_states, new_grads, adjoint_spikes = jax.lax.cond(
        no_event,
        lambda *args: (adjoint_state, grads, adjoint_spikes),
        adjoint_tr_dynamics,
        *(
            adjoint_state,
            spike,
            adjoint_spike,
            queue_index,
            grads,
            weights,
            adjoint_spikes,
        ),
    )
    return (weights, adjoint_spikes, tr_states, new_grads)


def wrapped_adjoint_step(
    adjoint_step_fn: Callable,
    node: str,
    carry,
    step_idx: int,
):
    """
    Wrapper to run the adjoint step function for a specific layer and step.

    :param adjoint_step_fn: Function performing a single adjoint step.
    :param node: Name of the current layer/node.
    :param carry: Tuple containing
        (weights, spikes, queue_indices, adjoint_spikes,
         adjoint_states, grads).
    :param step_idx: Index of the current step in the scan.
    :returns: Updated carry tuple and dummy 0 (for scan compatibility).
    """
    adjoint_step_input = (
        carry[0],  # weights
        carry[1][node][step_idx],  # spike
        carry[2][step_idx],  # queue_index
        carry[3][node][step_idx],  # adjoint_spike
        carry[4],  # adjoint_state
        carry[5],  # grads
        carry[3],  # adjoint_spikes
    )

    res = adjoint_step_fn(adjoint_step_input)

    carry = (
        carry[0],  # weights
        carry[1],  # spikes
        carry[2],  # queue_indices
        res[1],  # adjoint_spikes
        res[2],  # adjoint_states
        res[3],  # grads
    )

    return carry, 0
