# pylint: disable=wrong-import-order,logging-not-lazy
# pylint: disable=logging-fstring-interpolation
"""
Defining basic types to create hw-executable instances
"""
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import _hxtorch_core
import _hxtorch_spiking
import hxtorch
import jax.numpy as np
import pygrenade_vx as grenade
from dlens_vx_v3 import hal, lola
from hxtorch.spiking.experiment import NeuronPlacement
from hxtorch.spiking.utils import calib_helper
from jaxsnn.base.types import Array
from jaxsnn.event.hardware.calib import WaferConfig
from jaxsnn.event.hardware.input_neuron import InputNeuron
from jaxsnn.event.hardware.neuron import Neuron
from jaxsnn.event.hardware.synapse import Synapse
from jaxsnn.event.types import Spike, Weight

log = logging.getLogger("root")

HardwareSpike = Dict[
    grenade.network.PopulationOnNetwork, Tuple[Array, Array]
]


class Experiment:
    """Experiment class for describing experiments on hardware"""

    # pyling: disable=too-many-arguments
    def __init__(
            self,
            wafer_config: WaferConfig,
            hw_routing_func=grenade.network.routing.PortfolioRouter(),
            execution_instance: grenade.common.ExecutionInstanceID
            = grenade.common.ExecutionInstanceID()) \
            -> None:
        """
        Instanziate a new experiment, represting an experiment on hardware
        and/or in software.
        """

        # Load chip object
        self.wafer_config = wafer_config
        self._chip = self.load_calib(wafer_config.file)

        # Grenade stuff
        self.grenade_network = None
        self.grenade_network_graph = None

        self._populations: List[Union[InputNeuron, Neuron]] = []
        self._projections: List[Synapse] = []

        self._batch_size = 0

        self.id_counter = 0

        self.neuron_placement = NeuronPlacement()
        self.hw_routing_func = hw_routing_func
        self.execution_instance = execution_instance
        self._static_config_prepared = False
        self._populations_configured = False
        self.has_madc_recording = False

    def clear(self) -> None:
        """
        Reset the experiments's state. Corresponds to creating a new Experiment
        instance.
        """
        self.grenade_network = None
        self.grenade_network_graph = None

        self._chip = None
        self.wafer_config = None
        self._static_config_prepared = False

        self._populations = []
        self._projections = []

        self._batch_size = 0
        self.id_counter = 0

        self._populations_configured = False
        self.has_madc_recording = False

    def _prepare_static_config(self) -> None:
        """
        Prepares all the static chip config. Accesses the chip object
        initialized by hxtorch.hardware_init and appends corresponding
        configurations to. Additionally this method defines the
        pre_static_config builder injected to grenade at run.
        """
        if self._static_config_prepared:  # Only do this once
            return

        # If chip is still None we load default nightly calib
        if self._chip is None:
            log.info("No chip present. Using chip with default nightly calib.")
            self._chip = self.load_calib(calib_helper.nightly_calib_path())

        self._static_config_prepared = True
        log.debug("Preparation of static config done.")

    def load_calib(
        self, calib_path: Optional[Union[Path, str]] = None
    ) -> lola.Chip:
        """
        Load a calibration from path `calib_path` and apply to the experiment`s
        chip object. If no path is specified a nightly calib is applied.
        :param calib_path: The path to the calibration. It None, the nightly
            calib is loaded.
        :return: Returns the chip object for the given calibration.
        """
        # If no calib path is given we load spiking nightly calib
        log.info(f"Loading calibration from {calib_path}")
        self._chip = calib_helper.chip_from_file(calib_path)
        return self._chip

    def _generate_network_graphs(
        self,
        weights: List[Weight],
        build_graph: bool = True,
    ) -> grenade.network.NetworkGraph:
        """
        Generate grenade network graph from the populations and projections in
        modules

        :return: Returns the grenade network graph.
        """
        if not build_graph:
            assert self.grenade_network_graph is not None
            return self.grenade_network_graph

        # Create network builder
        network_builder = grenade.network.NetworkBuilder()

        # add input population
        for pop in self._populations:
            # log.info(f"Adding population of size: {pop.size}")
            pop.add_to_network_graph(network_builder)

        # for i, weight in enumerate(weights):
        #     synapse = Synapse(self, weight.input.T)
        #     synapse.add_to_network_graph(
        #         network_builder,
        #         self._populations[i].descriptor,
        #         self._populations[1].descriptor,
        #         self.wafer_config.weight_scaling,
        #     )

        # first weights
        synapse = Synapse(self, weights[0].input[:, :100].T)
        synapse.add_to_network_graph(
            network_builder,
            self._populations[0].descriptor,
            self._populations[1].descriptor,
            self.wafer_config.weight_scaling,
        )

        # second weights
        synapse = Synapse(self, weights[0].recurrent[:100, 100:103].T)
        synapse.add_to_network_graph(
            network_builder,
            self._populations[1].descriptor,
            self._populations[2].descriptor,
            self.wafer_config.weight_scaling,
        )
        network = network_builder.done()

        # route network if required
        routing_result = None
        if self.grenade_network_graph is None \
                or grenade.network.requires_routing(
                    network, self.grenade_network_graph):
            routing_result = self.hw_routing_func(network)

        # Keep graph
        self.grenade_network = network

        # build or update network graph
        if routing_result is not None:
            self.grenade_network_graph = grenade.network\
                .build_network_graph(
                    self.grenade_network, routing_result)
        else:
            grenade.network.update_network_graph(
                self.grenade_network_graph,
                self.grenade_network)

        return self.grenade_network_graph

    def _configure_populations(self):
        """
        Configure the population on hardware.
        """
        # Make sure experiment holds chip config
        assert self._chip is not None
        if self._populations_configured:
            return

        for module in self._populations:
            if not isinstance(module, Neuron):
                continue

            log.debug(f"Configure population '{module}'.")
            for in_pop_id, unit_id in enumerate(module.unit_ids):
                coord = self.neuron_placement.id2logicalneuron(unit_id)
                self._chip.neuron_block = module.configure_hw_entity(
                    in_pop_id, self._chip.neuron_block, coord
                )
                log.debug(f"Configured neuron at coord {coord}.")
        self._populations_configured = True

    def _generate_inputs(
        self,
        network_graph: grenade.network.NetworkGraph,
        inputs: Spike,
    ) -> grenade.signal_flow.IODataMap:
        """
        Generate external input events from the routed network graph
        representation.
        """
        assert network_graph.graph_translation.execution_instances[
            self.execution_instance].event_input_vertex is not None
        if network_graph.graph_translation.execution_instances[
                self.execution_instance].event_input_vertex is None:
            return grenade.signal_flow.IODataMap()

        self._batch_size = inputs.time.shape[0]
        input_generator = grenade.network.InputGenerator(
            network_graph, self._batch_size
        )

        assert len(self._populations) > 0, "no population registered"
        input_pop = self._populations[0]
        assert isinstance(
            input_pop, InputNeuron
        ), "need input neuron as first population"

        input_pop.add_to_input_generator(inputs, input_generator)
        return input_generator.done()

    def _get_population_observables(
        self,
        network_graph: grenade.network.NetworkGraph,
        result_map: grenade.signal_flow.IODataMap,
        runtime,
        n_spikes: List[int],
    ) -> HardwareSpike:
        """
        Takes the greade network graph and the result map returned by grenade
        after experiment execution and returns a data map where for each
        population descriptor of registered populations the population-specific
        hardware observables are represented as Optional[torch.Tensor]s.
        Note: This function calles the modules `post_process` method.
        :param network_graph: The logical grenade network graph describing the
            logic of the experiment.
        :param result_map: The result map returned by grenade holding all
            recorded hardware observables.
        :param runtime: The runtime of the experiment executed on hardware in
            clock cycles.
        :returns: Returns the data map as dict, where the keys are the
            population descriptors and values are tuples of values returned by
            the correpsonding module's `post_process` method.
        """
        # Get hw data
        assert len(n_spikes) == len(self._populations[1:])
        n_spike_map = {
            p.descriptor: n for n, p in zip(n_spikes, self._populations[1:])
        }
        return _hxtorch_core.extract_n_spikes(
            result_map, network_graph, runtime, n_spike_map
        )

    def register_population(self, module: Union[InputNeuron, Neuron]) -> None:
        """
        Register a module as population.

        :param module: The module to register as population.
        """
        self._populations.append(module)

    def register_projection(self, module: Synapse) -> None:
        """
        Register a module as projection.

        :param module: The module to register as projection.
        """
        self._projections.append(module)

    def get_hw_results(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        inputs: Spike,
        weights: List[Weight],
        runtime: int,
        n_spikes: List[int],
        time_data: Optional[dict] = None,
        build_graph: bool = True,
        hw_cycle_correction: Optional[int] = None,
    ) -> List[Spike]:
        """
        Executes the experiment on hardware using the information
        added to the experiment for a time given by `runtime` and returns a
        dict of hardware data represented as PyTorch data types.

        :param runtime: The runtime of the experiment on hardware in us.

        :returns: Returns the data map as dict, where the keys are the
            population descriptors and values are tuples of values returned by
            the correpsonding module's `post_process` method.
        """
        overall_start = time.time()
        self._prepare_static_config()

        # Generate network graph
        start = time.time()
        network = self._generate_network_graphs(
            weights, build_graph=build_graph
        )
        generate_network_time = time.time() - start

        # configure populations
        self._configure_populations()

        # handle runtime
        cycles_per_us = int(hal.Timer.Value.fpga_clock_cycles_per_us)
        runtime_in_clocks = int(runtime * cycles_per_us)
        if runtime_in_clocks > hal.Timer.Value.max:
            max_runtime = hal.Timer.Value.max / cycles_per_us
            raise ValueError(
                f"Runtime of {runtime} too long. Maximum supported runtime "
                + f"{max_runtime}"
            )

        # generate external spike trains
        inputs = self._generate_inputs(network, inputs)
        inputs.runtime = [
            {grenade.common.ExecutionInstanceID(): runtime_in_clocks}
        ] * self._batch_size
        log.debug(f"Registered runtimes: {inputs.runtime}")

        grenade_start = time.time()
        outputs = _hxtorch_spiking.run(
            self._chip,
            network,
            inputs,
            grenade.signal_flow.ExecutionInstancePlaybackHooks()
        )
        time_grenade_run = time.time() - grenade_start
        start_get_observables = time.time()
        hw_data = self._get_population_observables(
            network, outputs, runtime_in_clocks, n_spikes)

        # convert from fpga cycles to s
        spike_list = []
        offset = self._populations[0].size
        for i in range(1, len(self._populations)):
            spikes = hw_data[self._populations[i].descriptor]
            spike_list.append(
                Spike(
                    idx=np.where(spikes[0] == -1, -1, spikes[0] + offset),
                    time=(spikes[1] + hw_cycle_correction)
                    / cycles_per_us
                    * 1e-6,
                )
            )
            offset += self._populations[i].size

        # madc
        if self.has_madc_recording:
            log.info(f"Runtime in clocks: {runtime_in_clocks}")
            hw_madc_samples = hxtorch.spiking.extract_madc(
                outputs, network, runtime_in_clocks
            )

            data = (
                hw_madc_samples[self._populations[1].descriptor]
                .data.to_dense()
                .numpy()
            )
            madc_recording = data[
                :,
                :,
                self._populations[  # pylint: disable=protected-access
                    1
                ]._record_neuron_id,  # pylint: disable=protected-access
            ]
            return spike_list, madc_recording

        # TODO #4038: drop this
        time_get_observables = time.time() - start_get_observables
        # save times
        if time_data is not None:
            time_get_hw_results = time.time() - overall_start
            if time_data.get("get_hw_results") is None:
                time_data["get_hw_results"] = time_get_hw_results
            else:
                time_data["get_hw_results"] += time_get_hw_results

            if time_data.get("get_observables") is None:
                time_data["get_observables"] = time_get_observables
            else:
                time_data["get_observables"] += time_get_observables

            if time_data.get("generate_network_time") is None:
                time_data["generate_network_time"] = generate_network_time
            else:
                time_data["generate_network_time"] += generate_network_time

            if time_data.get("grenade_run") is None:
                time_data["grenade_run"] = time_grenade_run
            else:
                time_data["grenade_run"] += time_grenade_run

        return spike_list, time_data
