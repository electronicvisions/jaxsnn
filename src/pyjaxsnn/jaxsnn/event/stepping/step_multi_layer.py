from typing import (
    List,
    Callable,
    Tuple,
    Dict,
)

from jaxsnn.event.types import (
    Observable,
    TrajectoryCarry,
    Step,
)


def multi_layer_step(  # pylint: disable=too-many-locals,too-many-arguments
    step_fns: Dict[str, Callable],
    nodes: List[str],
    node_index_mapping: Dict[str, int],
    carry: TrajectoryCarry,
    step_idx: int,
) -> Tuple[TrajectoryCarry, int]:
    """
    Perform one simulation step across multiple neuron layers.

    Iterates through all layers in the strongly connected component (SCC),
    applies their respective step functions, and updates spikes, states,
    queue heads, and queue indices accordingly.

    :param step_fns: Dictionary of step functions for each layer, keyed by node
        name.
    :param nodes: List of node names in the SCC to process.
    :param node_index_mapping: Dictionary mapping node names to their indices.
    :param carry: Tuple (parameters, spikes, states, queue_heads,
        queue_indices).
    :param step_idx: Current step index.

    :returns: Updated carry tuple and an integer placeholder (always 0).
    """

    for node in nodes:
        step_input = Step(
            carry.parameters,
            carry.observables,
            carry.external_spikes,
            carry.states[node],
            step_idx,
            node_index_mapping[node],
            carry.queue_heads[node],
            carry.queue_indices[node],
        )

        res = step_fns[node](step_input)
        new_spike, transitioned_state, this_queue_heads, this_queue_index = res

        carry.states[node] = transitioned_state
        carry.queue_heads[node] = this_queue_heads

        carry.observables[node] = Observable(
            spike=carry.observables[node].spike.set_item(
                key=step_idx,
                new_value=new_spike,
            ),
            rate=carry.observables[node].rate,
        )

        carry.queue_indices[node] = carry.queue_indices[node].at[step_idx].set(
            this_queue_index,
        )

    return carry, 0
