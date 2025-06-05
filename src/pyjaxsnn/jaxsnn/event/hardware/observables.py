""" Hardware observables object """
from dataclasses import dataclass

import _hxtorch_core
from hxtorch.core.observables import HXObservables


@dataclass
class Observables(HXObservables):

    def set_data(self, spikes=None, cadc=None, madc=None) -> None:
        """
        Set the data to be extracted. This method also evokes data extraction.

        :param network_graph: The logical grenade network graph describing the
            logic of the experiment.
        :param result_map: The result map returned by grenade holding all
            recorded hardware observables.
        """
        self.spikes = spikes
        self.cadc = cadc
        self.madc = madc

    def get_spikes(self, n_events: int, max_spikes: int):
        return _hxtorch_core.extract_n_spikes(
            self.spikes, n_events, max_spikes
        )
