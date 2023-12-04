import jax
from jaxsnn.base.params import LIFParameters
from jaxsnn.event.types import StepState, Weight, WeightInput, WeightRecurrent


def transition_with_recurrence(  # pylint: disable=too-many-arguments
    params: LIFParameters,
    state: StepState,
    weights: Weight,
    spike_idx: int,
    recurrent_spike: bool,
    prev_layer_start: int,
) -> StepState:
    def recurrent_transition(
        params: LIFParameters,
        state: StepState,
        weights: WeightRecurrent,
        spike_idx: int,
        prev_layer_start: int,  # pylint: disable=unused-argument
    ):
        tr_row = weights.recurrent[spike_idx]

        state.neuron_state.I = state.neuron_state.I + tr_row
        state.neuron_state.V = state.neuron_state.V.at[spike_idx].set(
            params.v_reset
        )
        return state

    def input_transition(
        params: LIFParameters,  # pylint: disable=unused-argument
        state: StepState,
        weights: WeightRecurrent,
        spike_idx: int,  # pylint: disable=unused-argument
        prev_layer_start: int,
    ):
        spike = state.input_queue.pop()
        index_for_layer = spike.idx - prev_layer_start
        input_previous_layer = index_for_layer >= 0
        state.neuron_state.I = jax.lax.cond(
            input_previous_layer,
            lambda: state.neuron_state.I + weights.input[index_for_layer],
            lambda: state.neuron_state.I,
        )
        return state

    return jax.lax.cond(
        recurrent_spike,
        recurrent_transition,
        input_transition,
        params,
        state,
        weights,
        spike_idx,
        prev_layer_start,
    )


def transition_without_recurrence(  # pylint: disable=too-many-arguments
    params: LIFParameters,
    state: StepState,
    weights: WeightInput,
    spike_idx: int,
    recurrent_spike: bool,
    prev_layer_start: int,
) -> StepState:
    def input_transition(
        state: StepState,
        weights: WeightInput,
        spike_idx: int,  # pylint: disable=unused-argument
        prev_layer_start: int,
    ):
        spike = state.input_queue.pop()
        index_for_layer = spike.idx - prev_layer_start
        input_previous_layer = index_for_layer >= 0
        state.neuron_state.I = jax.lax.cond(
            input_previous_layer,
            lambda: state.neuron_state.I + weights.input[index_for_layer],
            lambda: state.neuron_state.I,
        )
        return state

    def no_transition(
        state: StepState, *args
    ):  # pylint: disable=unused-argument
        state.neuron_state.V = state.neuron_state.V.at[spike_idx].set(
            params.v_reset
        )
        return state

    return jax.lax.cond(
        recurrent_spike,
        no_transition,
        input_transition,
        state,
        weights,
        spike_idx,
        prev_layer_start,
    )
