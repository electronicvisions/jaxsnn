from typing import Optional

import jax

from pyhalco_hicann_dls_vx_v3 import DLSGlobal

from jaxsnn.event.types import SourcePopulation
from jaxsnn.event.hardware.experiment import Experiment
from jaxsnn.event.hardware.modules.base_module import BaseModule
from jaxsnn.event.hardware.modules.input_neuron import InputNeuron


def HXSource(  # pylint: disable=invalid-name
    size: int,
    n_events: int,
    chip_coordinate: Optional[DLSGlobal] = None,
    enable_spike_loopback: bool = False,
) -> SourcePopulation:
    """
    Create an input layer descriptor.

    The init_fn, state_fn, and event_fn returned by the generator return None
    because they are not needed.

    :param size: Number of neurons in the input layer.
    :param n_events: Number of events to generate.
    :param chip_coordinate: Chip coordinate for hardware execution.
    :param enable_spike_loopback: Whether to enable spike loopback.

    :returns: SourcePopulation instance.
    """
    def generator(
        *args,  # pylint: disable=unused-argument
        **kwargs,  # pylint: disable=unused-argument
    ) -> SourcePopulation.Functions:
        """Generator function for the input layer."""

        def init_fn(
            rng: jax.Array,
        ) -> None:
            return None

        def state_fn() -> None:
            # TODO: Should set layer_idx to inputs because not known beforehand
            return None

        def event_fn(
            *args,  # pylint: disable=unused-argument
        ) -> None:
            return None

        def generate_hx_module_fn(
            layer_idx: int,
            experiment: Experiment,
            source: Optional[BaseModule] = None,  # pylint: disable=unused-argument
            target: Optional[BaseModule] = None,  # pylint: disable=unused-argument
        ) -> InputNeuron:
            return InputNeuron(
                layer_idx=layer_idx,
                size=size,
                n_events=n_events,
                experiment=experiment,
                chip_coordinate=chip_coordinate,
                enable_spike_loopback=enable_spike_loopback,
            )

        return SourcePopulation.Functions(
            init_fn,
            state_fn,
            event_fn,
            generate_hx_module_fn,
        )

    parameters = {
        "size": size,
        "chip_coordinate": chip_coordinate,
        "enable_spike_loopback": enable_spike_loopback,
    }

    return SourcePopulation(generator, parameters, size)
