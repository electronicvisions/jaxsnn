from typing import Tuple, Dict, Union

import jax
import jax.numpy as jnp

from jaxsnn.base.types import Parameter
from jaxsnn.event.types import StepState, Observables


def add_grads(
    tau_syn: Union[float, jax.Array],
    pre_pop_node: str,
    pre_param_node: str,
    grads: Dict[str, Parameter],
    index_for_layer: int,
    adjoint_states: StepState,
    weights: jax.Array,
    adjoint_observables: Observables,
    queue_index: int,
) -> Tuple[Dict[str, Parameter], Observables]:
    """
    Add gradients with respect to synaptic weights and update adjoint spikes
    time for backpropagating the gradient.

    :param tau_syn: Synaptic time constant.
    :param pre_pop_node: Name of the pre-synaptic population node.
    :param pre_param_node: Name of the parameter node associated with the
        connection.
    :param grads: Dictionary containing gradients with respect to the
        parameters.
    :param index_for_layer: Index of the pre-synaptic neuron.
    :param adjoint_states: Adjoint states of the post-synaptic layer.
    :param weights: Dictionary containing the weights.
    :param adjoint_spikes: Adjoint spikes structure used for backpropagation.
    :param queue_index: Index in the spike queue for the input event.

    :returns: Updated grads and adjoint_spikes.
    """

    new_grad = grads[pre_param_node].at[
        index_for_layer
    ].add(
        - tau_syn * adjoint_states.neuron_state.I
    )

    new_term = jnp.dot(
        weights[pre_param_node][index_for_layer],
        (
            adjoint_states.neuron_state.V
            - adjoint_states.neuron_state.I
        ),
    )
    new_spike_time = \
        adjoint_observables[pre_pop_node].spike.time.at[queue_index].add(
            new_term
        )

    new_grads = {**grads, pre_param_node: new_grad}
    new_adj_observables = jax.tree_util.tree_map(
        lambda x: x,
        adjoint_observables
    )
    new_adj_observables[pre_pop_node].spike.time = new_spike_time

    return new_grads, new_adj_observables
