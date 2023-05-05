from .module import Module
import time
import pygrenade_vx.network.placed_logical as grenade
from jaxsnn.base.types import Spike
import hxtorch
from jaxsnn.event.hardware import utils
import numpy as onp
from jaxsnn.event.leaky_integrate_and_fire import LIFParameters
import _hxtorch_core

log = hxtorch.logger.get("hxtorch.snn.modules")


class InputNeuron(Module):
    """
    Spike source generating spikes at the times [ms] given in the spike_times
    array.
    """

    def __init__(self, size: int, params: LIFParameters, experiment) -> None:
        """
        Instanziate a INputNeuron. This module serves as an External
        Population for input injection and is created within `snn.Experiment`
        if not present in the considerd model.
        This module performes an identity mapping when `forward` is called.

        :param size: Number of input neurons.
        :param experiment: Experiment to which this module is assigned.
        """
        super().__init__(experiment)
        self.size = size
        self.params = params
        self.register_hw_entity()

    def register_hw_entity(self) -> None:
        """
        Register instance in member `experiment`.
        """
        self.experiment.register_population(self)

    def add_to_network_graph(
        self, builder: grenade.NetworkBuilder
    ) -> grenade.PopulationDescriptor:
        """
        Adds instance to grenade's network builder.

        :param builder: Grenade network builder to add extrenal population to.
        :returns: External population descriptor.
        """
        # create grenade population
        population = grenade.ExternalPopulation(self.size)
        # add to builder
        self.descriptor = builder.add(population)
        log.TRACE(f"Added Input Population: {self}")

        return self.descriptor

    def add_to_input_generator(
        self, input: Spike, builder: grenade.InputGenerator
    ) -> None:
        """
        Add the neurons events represented by this instance to grenades input
        generator.

        :param input: input spikes for this neuron
        :param builder: Grenade's input generator to append the events to.
        """
        # convert input from seconds to milliseconds
        spike_tuple = (onp.array(input.idx), onp.array(input.time) * 1_000)
        spike_times = _hxtorch_core.dense_spikes_to_list(spike_tuple, self.size)
        builder.add(spike_times, self.descriptor)
