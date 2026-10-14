from dataclasses import replace
from typing import Tuple, List, Union

import jax
from jax import numpy as jnp
from jaxsnn.event import custom_lax
from jaxsnn.event.types import (
    AddGradFn,
    Observable,
    Observables,
    QueueIndex,
    StepState,
    Parameters,
)


def adjoint_transition(  # pylint: disable=too-many-arguments
    v_threshold: Union[jax.Array, float],
    v_reset: Union[jax.Array, float],
    input_layers: List[int],
    add_grads_fns: List[AddGradFn],
    adjoint_states: StepState,
    observable: Observable,
    adjoint_observable: Observable,
    queue_index: QueueIndex,
    params_grad: Parameters,
    params: Parameters,
    adjoint_observables: Observables,
) -> Tuple[StepState, Parameters, Observables]:
    """
    Perform the adjoint transition step for a neuron or input spike.

    :param v_threshold: Membrane threshold potential.
    :param v_reset: Membrane reset potential.
    :param input_layers: List of input layer indices.
    :param add_grads_fns: List of functions to add gradients for weights.
    :param adjoint_states: Current adjoint neuron states.
    :param observable: Current spike event.
    :param adjoint_observable: Current Adjoint spike event.
    :param queue_index: Index of the spike in the queue.
    :param grads: Gradients for weights.
    :param weights: Weights of the current layer.
    :param adjoint_observables: Adjoint spikes through which the gradient flows.

    :returns: Updated adjoint_states, grads, and adjoint_observables.
    """

    def internal_transition(
        adjoint_states,
        grads,
        adjoint_spikes,
    ):
        epsilon = 1e-6
        safe_denominator = jnp.where(
            jnp.abs(observable.spike.current - v_threshold) > epsilon,
            observable.spike.current - v_threshold,
            epsilon,
        )

        transitioned_V = adjoint_states.neuron_state.V.at[
            observable.spike.idx
        ].add(
            (
                adjoint_observable.spike.time
                + (v_threshold - v_reset)
                * adjoint_states.neuron_state.V[observable.spike.idx]
            )
            / safe_denominator
            - adjoint_observable.rate[observable.spike.idx]
        )

        adjoint_states = replace(
            adjoint_states,
            neuron_state=replace(
                adjoint_states.neuron_state,
                V=transitioned_V
            ),
        )

        return adjoint_states, grads, adjoint_spikes

    def input_transition(
        adjoint_states,
        grads,
        adjoint_spikes,
    ):
        rel_input_layer_idx = jnp.argmin(
            jnp.where(input_layers == observable.spike.layer_idx, 0, 1)
        )

        grads, adjoint_spikes = custom_lax.switch(
            rel_input_layer_idx,
            add_grads_fns,
            grads,
            observable.spike.idx,
            adjoint_states,
            params,
            adjoint_observables,
            queue_index,
        )

        return adjoint_states, grads, adjoint_spikes

    return custom_lax.cond(
        observable.spike.internal,
        internal_transition,
        input_transition,
        adjoint_states,
        params_grad,
        adjoint_observables,
    )
