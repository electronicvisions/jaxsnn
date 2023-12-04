# pylint: disable=invalid-name
from typing import Callable

import jax
import jax.numpy as np
from jaxsnn.base.params import LIFParameters
from jaxsnn.event.flow import exponential_flow
from jaxsnn.event.types import (
    EventPropSpike,
    LIFState,
    StepState,
    Weight,
    WeightInput,
    WeightRecurrent,
)


def adjoint_transition_without_recurrence(  # pylint: disable=too-many-arguments
    params: LIFParameters,
    adjoint_state: StepState,
    spike: EventPropSpike,
    layer_start: int,
    adjoint_spike: EventPropSpike,
    grads: Weight,
    weights: Weight,
    input_queue_head: int,
):
    def adjoint_transition_in_layer(  # pylint: disable=too-many-arguments,unused-argument
        params: LIFParameters,
        adjoint_state: StepState,
        spike: EventPropSpike,
        layer_start: int,
        adjoint_spike: EventPropSpike,
        grads: Weight,
        weights: Weight,
        input_queue_head: int,
    ):
        epsilon = 1e-6
        safe_denominator = np.where(
            np.abs(spike.current - params.v_th) > epsilon,
            spike.current - params.v_th,
            epsilon,
        )
        adjoint_state.neuron_state.V = adjoint_state.neuron_state.V.at[
            spike.idx - layer_start
        ].add(adjoint_spike.time / safe_denominator)
        return adjoint_state, grads

    def adjoint_input_transition(  # pylint: disable=too-many-arguments,unused-argument
        params: LIFParameters,
        adjoint_state: StepState,
        spike: EventPropSpike,
        layer_start: int,
        adjoint_spike: EventPropSpike,
        grads: WeightInput,
        weights: WeightInput,
        input_queue_head: int,
    ):
        prev_layer_start = layer_start - weights.input.shape[0]
        index_for_layer = spike.idx - prev_layer_start

        # do nothing if spike is not from directly previous layer
        grads, dt = jax.lax.cond(
            index_for_layer >= 0,
            lambda: (
                WeightInput(
                    grads.input.at[index_for_layer].add(
                        -params.tau_syn * adjoint_state.neuron_state.I
                    ),
                ),
                np.dot(
                    weights.input[index_for_layer],
                    (
                        adjoint_state.neuron_state.V
                        - adjoint_state.neuron_state.I
                    ),
                ),
            ),
            lambda: (grads, 0.0),
        )
        adjoint_state.input_queue.spikes.time = (
            adjoint_state.input_queue.spikes.time.at[input_queue_head - 1].set(
                dt
            )
        )
        adjoint_state.input_queue.head += 1
        return adjoint_state, grads

    spike_in_layer = spike.idx >= layer_start
    return jax.lax.cond(
        spike_in_layer,
        adjoint_transition_in_layer,
        adjoint_input_transition,
        *(
            params,
            adjoint_state,
            spike,
            layer_start,
            adjoint_spike,
            grads,
            weights,
            input_queue_head,
        ),
    )


def adjoint_transition_with_recurrence(  # pylint: disable=too-many-arguments
    params: LIFParameters,
    adjoint_state: StepState,
    spike: EventPropSpike,
    layer_start: int,
    adjoint_spike: EventPropSpike,
    grads: Weight,
    weights: Weight,
    input_queue_head: int,
):
    def adjoint_transition_with_recurrence_in_layer(  # pylint: disable=too-many-arguments
        params: LIFParameters,
        adjoint_state: StepState,
        spike: EventPropSpike,
        layer_start: int,
        adjoint_spike: EventPropSpike,
        grads: WeightRecurrent,
        weights: WeightRecurrent,
        input_queue_head: int,  # pylint: disable=unused-argument
    ):
        index_for_layer = spike.idx - layer_start
        epsilon = 1e-6
        safe_denominator = np.where(
            np.abs(spike.current - params.v_th) > epsilon,
            spike.current - params.v_th,
            epsilon,
        )
        new_term = np.dot(
            weights.recurrent[index_for_layer, :],
            (adjoint_state.neuron_state.V - adjoint_state.neuron_state.I),
        )
        voltage = adjoint_state.neuron_state.V.at[index_for_layer].add(
            (adjoint_spike.time + new_term) / safe_denominator
        )
        updated_state = StepState(
            LIFState(voltage, adjoint_state.neuron_state.I),
            adjoint_state.time,
            adjoint_state.input_queue,
        )
        grads = WeightRecurrent(
            grads.input,
            grads.recurrent.at[index_for_layer].add(
                -params.tau_syn * updated_state.neuron_state.I
            ),
        )
        return updated_state, grads

    def adjoint_input_transition(  # pylint: disable=too-many-arguments
        params: LIFParameters,
        adjoint_state: StepState,
        spike: EventPropSpike,
        layer_start: int,
        adjoint_spike: EventPropSpike,  # pylint: disable=unused-argument
        grads: WeightRecurrent,
        weights: WeightRecurrent,
        input_queue_head: int,
    ):
        prev_layer_start = layer_start - weights.input.shape[0]
        index_for_layer = spike.idx - prev_layer_start

        grads = WeightRecurrent(
            grads.input.at[index_for_layer].add(
                -params.tau_syn * adjoint_state.neuron_state.I
            ),
            grads.recurrent,
        )
        dt = np.dot(
            weights.input[index_for_layer],
            (adjoint_state.neuron_state.V - adjoint_state.neuron_state.I),
        )

        adjoint_state.input_queue.spikes.time = (
            adjoint_state.input_queue.spikes.time.at[input_queue_head - 1].set(
                dt
            )
        )
        adjoint_state.input_queue.head += 1
        return adjoint_state, grads

    spike_in_layer = spike.idx >= layer_start
    updated_state, grads = jax.lax.cond(
        spike_in_layer,
        adjoint_transition_with_recurrence_in_layer,
        adjoint_input_transition,
        *(
            params,
            adjoint_state,
            spike,
            layer_start,
            adjoint_spike,
            grads,
            weights,
            input_queue_head,
        ),
    )

    return updated_state, grads


def adjoint_lif_exponential_flow(params: LIFParameters):
    A = np.array(
        [[-params.tau_mem_inv, 0.0], [params.tau_syn_inv, -params.tau_syn_inv]]
    )
    return exponential_flow(A)


def adjoint_lif_dynamic(params: LIFParameters, lambda_0: jax.Array, t: float):
    tau_exp = np.exp(-t / params.tau_mem)
    syn_exp = np.exp(-t / params.tau_syn)
    A = np.array(
        [
            [tau_exp, 0],
            [
                params.tau_mem
                / (params.tau_mem - params.tau_syn)
                * (tau_exp - syn_exp),
                tau_exp,
            ],
        ]
    )
    return np.dot(A, lambda_0)


# define hybrid adjoint dynamics (EventProp)
def step_bwd(  # pylint: disable=too-many-locals
    adjoint_dynamics: Callable,
    adjoint_tr_dynamics: Callable,
    t_max: float,
    res,
    g,
):
    (_, weights, layer_start), spike = res
    (adjoint_state, grads, _), adjoint_spike = g

    reversed_time = t_max - spike.time
    time_diff = reversed_time - adjoint_state.time

    # integrate lambdas to the spike
    adjoint_state.neuron_state = adjoint_dynamics(
        adjoint_state.neuron_state, time_diff
    )
    adjoint_state.time = reversed_time

    no_event = spike.idx == -1
    tr_state, new_grads = jax.lax.cond(
        no_event,
        lambda *args: (adjoint_state, grads),
        adjoint_tr_dynamics,
        *(
            adjoint_state,
            spike,
            layer_start,
            adjoint_spike,
            grads,
            weights,
            len(adjoint_state.input_queue.spikes.time)
            - adjoint_state.input_queue.head,
        ),
    )
    return (tr_state, new_grads, layer_start), 1
