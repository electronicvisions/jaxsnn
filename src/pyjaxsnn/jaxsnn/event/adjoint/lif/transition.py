from typing import Tuple, List, Union

import jax
from jax import numpy as jnp
from jaxsnn.event.types import (
    AddGradFn,
    QueueIndex,
    StepState,
    Spike,
    Parameters,
    IOData
)


def adjoint_transition(  # pylint: disable=too-many-arguments
    v_threshold: Union[jax.Array, float],
    v_reset: Union[jax.Array, float],
    input_layers: List[int],
    add_grads_fns: List[AddGradFn],
    adjoint_states: StepState,
    spike: Spike,
    adjoint_spike: Spike,
    queue_index: QueueIndex,
    grads: Parameters,
    weights: Parameters,
    adjoint_spikes: IOData,
) -> Tuple[StepState, Parameters, IOData]:
    """
    Perform the adjoint transition step for a neuron or input spike.

    :param v_threshold: Membrane threshold potential.
    :param v_reset: Membrane reset potential.
    :param input_layers: List of input layer indices.
    :param add_grads_fns: List of functions to add gradients for weights.
    :param adjoint_states: Current adjoint neuron states.
    :param spike: Current spike event.
    :param adjoint_spike: Current Adjoint spike event.
    :param queue_index: Index of the spike in the queue.
    :param grads: Gradients for weights.
    :param weights: Weights of the current layer.
    :param adjoint_spikes: Adjoint spikes through which the gradient flows.

    :returns: Updated adjoint_states, grads, and adjoint_spikes.
    """
    def adjoint_transition_in_layer(  # pylint: disable=too-many-arguments,unused-argument
        v_threshold: Union[float, jax.Array],
        v_reset: Union[float, jax.Array],
        adjoint_states: StepState,
        spike: Spike,
        adjoint_spike: Spike,
        grads: Parameters,
        weights: Parameters,
        adjoint_spikes: IOData,
        queue_index: int
    ):

        epsilon = 1e-6
        safe_denominator = jnp.where(
            jnp.abs(spike.current - v_threshold) > epsilon,
            spike.current - v_threshold,
            epsilon,
        )

        adjoint_states.neuron_state.V = adjoint_states.neuron_state.V.at[
            spike.idx
        ].add(
            (
                adjoint_spike.time
                + (v_threshold - v_reset) * adjoint_states.neuron_state.V[
                    spike.idx
                ]) / safe_denominator)

        return adjoint_states, grads, adjoint_spikes

    def adjoint_input_transition(  # pylint: disable=too-many-arguments,unused-argument
        v_threshold: Union[float, jax.Array],
        v_reset: Union[float, jax.Array],
        adjoint_states: StepState,
        spike: Spike,
        adjoint_spike: Spike,
        grads: Parameters,
        weights: Parameters,
        adjoint_spikes: IOData,
        queue_index: QueueIndex,
    ):

        rel_input_layer_idx = jnp.argmin(
            jnp.where(input_layers == spike.layer_idx, 0, 1))

        grads, adjoint_spikes = jax.lax.switch(
            rel_input_layer_idx,
            add_grads_fns,
            grads,
            spike.idx,
            adjoint_states,
            weights,
            adjoint_spikes,
            queue_index,
        )

        return adjoint_states, grads, adjoint_spikes

    return jax.lax.cond(
        spike.internal,
        adjoint_transition_in_layer,
        adjoint_input_transition,
        *(
            v_threshold,
            v_reset,
            adjoint_states,
            spike,
            adjoint_spike,
            grads,
            weights,
            adjoint_spikes,
            queue_index,
        ),
    )
