from typing import Union

import jax

from jaxsnn.event.types import StepState


def transition(  # pylint: disable=too-many-arguments
    v_reset: Union[float, jax.Array],
    input_node: str,
    state: StepState,
    weights: jax.Array,
    spike_idx: int,
    spike_in_layer: bool,
) -> StepState:
    """
    Apply state transition based on internal/input spike.

    :param v_reset: Reset potential for the neuron.
    :param input_node: Name of the input node/layer.
    :param state: Current step state containing neuron states.
    :param weights: Synaptic weight matrix.
    :param spike_idx: Index of the neuron that spiked.
    :param spike_in_layer: Boolean indicating if spike is internal/input.

    :returns: Updated step state after transition.
    """
    def input_transition(
        state: StepState,
        weights: jax.Array,
        spike_idx: int,  # pylint: disable=unused-argument
    ) -> StepState:

        state.neuron_state.I = (
            state.neuron_state.I + weights[input_node][spike_idx]
        )

        return state

    def no_transition(
        state: StepState, *args
    ) -> StepState:  # pylint: disable=unused-argument
        state.neuron_state.V = state.neuron_state.V.at[spike_idx].set(
            v_reset
        )
        return state

    return jax.lax.cond(
        spike_in_layer,
        no_transition,
        input_transition,
        state,
        weights,
        spike_idx,
    )
