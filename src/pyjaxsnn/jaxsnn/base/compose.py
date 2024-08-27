from typing import List, Tuple, Optional, Any

import jax
from jaxsnn.event.types import (
    EventPropSpike,
    InitApply,
    SingleInitApply,
    Weight,
)


def serial(*layers: SingleInitApply) -> InitApply:
    """Concatenate multiple layers of init/apply functions

    Returns:
        InitApply: Init/apply pair
    """
    init_fns, apply_fns = zip(*layers)

    def init_fn(
        rng: jax.random.KeyArray,
        input_size: int,
    ) -> Tuple[int, List[Weight]]:
        """Iterate and call the individual init functions"""
        weights = []

        for layer_init_fn in init_fns:
            rng, input_size, layer_params = layer_init_fn(rng, input_size)
            # Ensure that list of weights stays flat
            if isinstance(layer_params, List):
                weights.extend(layer_params)
            else:
                weights.append(layer_params)
        return input_size, weights

    def apply_fn(
        weights: List[Weight],
        spikes: EventPropSpike,
        external: Optional[Any] = None,
        carry: Optional[Any] = None,
    ) -> Tuple[Any, List[Weight], EventPropSpike, List[EventPropSpike]]:
        """Forward function of the network.

        Take parameters of the network and the input spikes and return the
        output spikes of each layer.

        Args:
            layer_start (int): Index of first neuron from layer
            weights (List[Weight]): Parameters of the network
            spikes (EventPropSpike): Input spikes
            recording (List[EventPropSpike]): Spikes of each layer

        Returns:
            List[EventPropSpike]: Spikes of each layer
        """

        recording = []
        future_weights = []
        for layer_apply_fn in apply_fns:
            layer_result = layer_apply_fn(
                weights, spikes, external, carry
            )
            carry, layer_future_params, spikes, layer_recording = layer_result
            # Ensure that weights and recording stay flat
            if isinstance(layer_recording, list):
                future_weights.extend(layer_future_params)
                recording.extend(layer_recording)
            else:
                future_weights.append(layer_future_params)
                recording.append(layer_recording)
        return carry, future_weights, spikes, recording

    return init_fn, apply_fn
