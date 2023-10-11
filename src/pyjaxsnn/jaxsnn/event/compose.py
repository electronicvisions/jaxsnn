from typing import List

import jax
from jaxsnn.event.types import (
    EventPropSpike,
    InitApply,
    InitApplyHW,
    SingleInitApply,
    SingleInitApplyHW,
    Weight,
    Spike,
)


def serial(*layers: SingleInitApply) -> InitApply:
    """Concatenate multiple layers of init/apply functions

    Returns:
        InitApply: Init/apply pair
    """
    # init_fns, apply_fns = zip(*layers)
    init_fns = [l[0] for l in layers]
    apply_fns = [l[1] for l in layers]

    def init_fn(rng: jax.random.KeyArray, input_shape: int) -> List[Weight]:
        """Iterate and call the individual init functions"""
        weights = []
        for init_fn in init_fns:
            if len(init_fns) > 1:
                rng, layer_rng = jax.random.split(rng)
            else:
                layer_rng = rng
            input_shape, param = init_fn(layer_rng, input_shape)
            weights.append(param)
        return weights

    def apply_fn(
        weights: List[Weight], spikes: EventPropSpike
    ) -> List[EventPropSpike]:
        """Take parameters of the network and the input spikes and return the output spikes of each layer

        Args:
            weights (List[Weight]): Parameters of the network
            spikes (EventPropSpike): Input spikes

        Returns:
            List[EventPropSpike]: Spikes of each layer
        """
        recording = []
        layer_start = 0
        for fn, param in zip(apply_fns, weights):
            layer_start += param.input.shape[0]
            spikes = fn(layer_start, param, spikes)
            recording.append(spikes)
        return recording

    return init_fn, apply_fn


def serial_spikes_known(*layers: SingleInitApplyHW) -> InitApplyHW:
    """Concatenate multiple layers of init/apply functions for the special case of the spikes already known.

    For a special case of hardware-in-the-loop training it is necessary to do a forward run in software (after already
    having the observations from software) in order to add information about the synaptic current at spike time.
    This information is not returned from the hardware but needed for the EventProp algorithm. As there is one more
    input, a different concatenation functino is needed.

    Returns:
        InitApply: Init/apply pair
    """
    # init_fns, apply_fns = zip(*layers)
    init_fns = [l[0] for l in layers]
    apply_fns = [l[1] for l in layers]

    def init_fn(rng: jax.random.KeyArray, input_shape: int) -> List[Weight]:
        """Iterate and call the individual init functions"""
        weights = []
        for init_fn in init_fns:
            if len(init_fns) > 1:
                rng, layer_rng = jax.random.split(rng)
            else:
                layer_rng = rng
            input_shape, param = init_fn(layer_rng, input_shape)
            weights.append(param)
        return weights

    def apply_fn(
        known_spikes: List[Spike],
        weights: List[Weight],
        spikes: EventPropSpike,
    ) -> List[EventPropSpike]:
        """Take parameters of the network and the input spikes and return the output spikes of each layer

        Args:
            known_spikes(List[EventPropSpike]): The spikes that happened on hardware in each layer
            weights (List[Weight]): Parameters of the network
            spikes (EventPropSpike): Input spikes

        Returns:
            List[EventPropSpike]: Spikes of each layer
        """
        recording = []
        layer_start = 0
        for fn, param, known in zip(apply_fns, weights, known_spikes):
            layer_start += param.input.shape[0]
            spikes = fn(layer_start, param, spikes, known)
            recording.append(spikes)
        return recording

    return init_fn, apply_fn
