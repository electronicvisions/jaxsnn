from __future__ import annotations
from abc import abstractmethod
import re
from typing import Optional

from hxtorch.core.modules.population import Population as CorePopulation
from hxtorch.core.modules.population import (
    InputPopulation as CoreInputPopulation
)

from jaxsnn.event.hardware.modules.base_module import BaseModule
from jaxsnn.event.hardware.experiment import Experiment
from jaxsnn.event.hardware.observables import Observables
from jaxsnn.event.types import Spike

from pyhalco_hicann_dls_vx_v3 import DLSGlobal


class InputPopulation(CoreInputPopulation, BaseModule):
    """ Input neuron class """
    experiment: Experiment
    _observables_factory = Observables

    def __init__(
        self,
        layer_idx: int,
        size: int,
        n_events: int,
        experiment: Experiment,
        chip_coordinate: Optional[DLSGlobal] = None,
        enable_spike_loopback: bool = False
    ):
        CoreInputPopulation.__init__(
            self,
            size,
            experiment,
            chip_coordinate,
            enable_spike_loopback,
        )
        BaseModule.__init__(
            self,
            layer_idx,
        )
        self.n_events = n_events

    @abstractmethod
    def set_spike_times(self, input_spikes: Spike) -> None:
        """ """

    @property
    def expected_return_type(self) -> Spike:
        """
        Returns the expected return type of the neuron.
        """
        # if self.enable_spike_loopback:
        batch_size = self.experiment.batch_size
        return Spike.empty((batch_size, self.n_events))


class Population(CorePopulation, BaseModule):
    """ Base class for all populations """
    experiment: Experiment
    _observables_factory = Observables

    def __init__(
        self,
        layer_idx: int,
        n_events: int,
        n_hw_spikes: Optional[int],
        *args,
        time_offset: float,
        **kwargs,
    ):
        CorePopulation.__init__(
            self,
            *args,
            **kwargs
        )
        BaseModule.__init__(
            self,
            layer_idx,
        )
        self.n_events = n_events
        self.time_offset = time_offset
        if n_hw_spikes is None:
            self.n_hw_spikes = n_events
        else:
            self.n_hw_spikes = n_hw_spikes

    @property
    def expected_return_type(self) -> Spike:
        """
        Returns the expected return type of the neuron.
        """
        batch_size = self.experiment.batch_size
        return Spike.empty((batch_size, self.n_events))
