# pylint: disable=wrong-import-order,logging-not-lazy,logging-fstring-interpolation
"""
Implementing SNN modules
"""
import logging
from typing import List, Optional

import numpy as np
import pygrenade_vx.network as grenade
from dlens_vx_v3 import hal, halco, lola
from hxtorch.spiking.morphology import SingleCompartmentNeuron
from jaxsnn.event.hardware.module import Module
from jaxsnn.event.leaky_integrate_and_fire import LIFParameters

log = logging.getLogger("root")


class Neuron(Module):
    """
    Neuron layer

    Caveat:
    For execution on hardware, this module can only be used in conjuction with
    a preceding Synapse module.
    """

    _madc_readout_source: hal.NeuronConfig.ReadoutSource = (
        hal.NeuronConfig.ReadoutSource.membrane
    )

    def __init__(  # pylint:disable=too-many-arguments
        self,
        size: int,
        params: LIFParameters,
        experiment,
        placement_constraint: Optional[List[halco.LogicalNeuronOnDLS]] = None,
        enable_madc_recording: bool = False,
        record_neuron_id: Optional[int] = None,
    ) -> None:
        """
        Initialize a Neuron. This module creates a population of spiking
        neurons of size `size`.

        :param size: Size of the population.
        :param experiment: Experiment to append layer to.
        :param placement_constraint: An optional list of logical neurons
            defining where to place the module`s neurons on hardware.
        :param enable_madc_recording: Enables or disables the recording of the
            neurons `record_neuron_id` membrane trace via the MADC. Only a
            single neuron can be recorded. This membrane traces is samples with
            a significant higher resolution as with the CADC.
        :param record_neuron_id: The in-population neuron index of the neuron
            to be recorded with the MADC. This has only an effect when
            `enable_madc_recording` is enabled.
        """
        super().__init__(experiment=experiment)

        self.size = size
        self.params = params
        self._placement_constraint = placement_constraint
        self.unit_ids: Optional[np.ndarray] = None

        self._neuron_structure = SingleCompartmentNeuron(1)
        self._enable_madc_recording = enable_madc_recording
        self._record_neuron_id = record_neuron_id
        self.register_hw_entity()

    def register_hw_entity(self) -> None:
        """
        Infere neuron ids on hardware and register them.
        """
        self.unit_ids = np.arange(
            self.experiment.id_counter, self.experiment.id_counter + self.size
        )
        self.experiment.neuron_placement.register_id(
            self.unit_ids,
            self._neuron_structure.compartments,
            self._placement_constraint,
        )
        self.experiment.id_counter += self.size
        self.experiment.register_population(self)
        if self._enable_madc_recording:
            if self.experiment.has_madc_recording:
                raise RuntimeError(
                    "Another HXModule already registered MADC recording. "
                    + "MADC recording is only enabled for a "
                    + "single neuron."
                )
            self.experiment.has_madc_recording = True
        log.debug(f"Registered hardware  entity '{self}'.")

    def configure_hw_entity(
        self,
        neuron_id: int,
        neuron_block: lola.NeuronBlock,
        coord: halco.LogicalNeuronOnDLS,
    ) -> lola.NeuronBlock:
        """
        Configures a neuron in the given layer with its specific properties.
        The neurons digital event outputs are enabled according to the given
        spiking mask.

        :param neuron_id: In-population neuron index.
        :param neuron_block: The neuron block hardware entity.
        :param coord: Coordinate of neuron on hardware.
        :returns: Configured neuron block.
        """
        self._neuron_structure.implement_morphology(coord, neuron_block)
        self._neuron_structure.set_spike_recording(True, coord, neuron_block)
        if neuron_id == self._record_neuron_id:
            log.info(f"Configuring madc recording for neuron {neuron_id}")
            self._neuron_structure.enable_madc_recording(
                coord, neuron_block, self._madc_readout_source
            )
        return neuron_block

    def add_to_network_graph(
        self, builder: grenade.NetworkBuilder
    ) -> grenade.PopulationOnNetwork:
        """
        Add the layer's neurons to grenades network builder.
        Note, the event output of the neurons
        are configured in `configure_hw_entity`.

        :param builder: Grenade's network builder to add the layer's population
            to.
        :returns: Returns the builder with the population added.
        """
        # get neuron coordinates
        coords: List[
            halco.LogicalNeuronOnDLS
        ] = self.experiment.neuron_placement.id2logicalneuron(self.unit_ids)

        # create receptors
        receptors = set(
            [
                grenade.Receptor(
                    grenade.Receptor.ID(), grenade.Receptor.Type.excitatory
                ),
                grenade.Receptor(
                    grenade.Receptor.ID(), grenade.Receptor.Type.inhibitory
                ),
            ]
        )

        neurons: List[grenade.Population.Neuron] = [
            grenade.Population.Neuron(
                logical_neuron,
                {
                    halco.CompartmentOnLogicalNeuron(): grenade.Population.Neuron.Compartment(  # pylint: disable=line-too-long
                        grenade.Population.Neuron.Compartment.SpikeMaster(
                            0, True
                        ),
                        [receptors] * len(logical_neuron.get_atomic_neurons()),
                    )
                },
            )
            for logical_neuron in coords
        ]

        # create grenade population
        gpopulation = grenade.Population(neurons)

        # add to builder
        self.descriptor = builder.add(gpopulation)

        # No recording registered -> return
        if not self._enable_madc_recording:
            return self.descriptor

        # add MADC recording
        madc_recording = grenade.MADCRecording()
        madc_recording.population = self.descriptor
        madc_recording.source = self._madc_readout_source
        madc_recording.neuron_on_population = int(self._record_neuron_id)
        madc_recording.compartment_on_neuron = (
            halco.CompartmentOnLogicalNeuron()
        )
        madc_recording.atomic_neuron_on_compartment = 0
        builder.add(madc_recording)
        log.debug(f"Added population '{self}' to grenade graph.")
        return self.descriptor
