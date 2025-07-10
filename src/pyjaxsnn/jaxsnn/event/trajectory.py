from typing import Callable, Tuple, Any, Optional

import jax
import jax.numpy as jnp
from jaxsnn.event.types import (
    EventPropSpike, InputQueue, LIFState, SingleApply, StepState, Weight)
from jaxsnn.event.stepping.types import StepInput
from jaxsnn.event.utils.filter import filter_spikes


def trajectory(
    step_fn: Callable[[StepInput, int], Tuple[StepInput, EventPropSpike]],
    size: int,
    n_spikes: int,
) -> SingleApply:
    """Evaluate the `step_fn` until `n_spikes` have been simulated.

    Uses a scan over the `step_fn` to return an apply function
    """

    def apply_fn(  # pylint: disable=unused-argument
        weights: Weight,
        input_spikes: EventPropSpike,
        external: Any,
        carry: Optional[int],
    ) -> Tuple[int, Weight, EventPropSpike, EventPropSpike]:
        if carry is None:
            layer_index = 0
            layer_start = 0
        else:
            layer_index, layer_start = carry
        this_layer_weights = weights[layer_index]
        input_size = this_layer_weights.input.shape[0]
        layer_start = layer_start + input_size

        # Filter out input spikes which are not from previous layer
        input_spikes = filter_spikes(input_spikes, layer_start - input_size)

        step_state = StepState(
            neuron_state=LIFState(
                jnp.zeros(size), jnp.zeros(size)),
            spike_times=-1 * jnp.ones(size),
            spike_mask=jnp.zeros(size, dtype=bool),
            time=0.0,
            input_queue=InputQueue(input_spikes),
        )
        _, spikes = jax.lax.scan(
            step_fn,
            (step_state, this_layer_weights, layer_start),
            jnp.arange(n_spikes)
        )

        layer_index += 1
        return (layer_index, layer_start), this_layer_weights, spikes, spikes

    return apply_fn
