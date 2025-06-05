from typing import Optional

import numpy as np

import _hxtorch_core

from jaxsnn.event.hardware.modules.population import InputPopulation
from jaxsnn.event.types import Spike


class InputNeuron(InputPopulation):
    """
    Spike source generating spikes at the times [ms] given in the spike_times
    array.
    """
    def set_spike_times(self, spike_times: Spike) -> None:
        """
        Set the spike times for this neuron.

        :param spike_times: Array of spike times in s.
        """
        self.input_spikes = spike_times
        self.changed_input_data = True

    def get_spike_times(self) -> None:
        """
        Add the neurons events represented by this instance to grenades input
        generator.

        :param inputs: input spikes for this neuron
        :param builder: Grenade's input generator to append the events to.
        """
        # convert input from seconds to ms
        spike_tuple = (
            np.array(self.input_spikes.idx),
            np.array(self.input_spikes.time) * 1E6
        )
        spike_times = _hxtorch_core.dense_spikes_to_list(spike_tuple, self.size)
        return spike_times

    def get_post_processed(self) -> Optional[Spike]:
        """ """
        if self.enable_spike_loopback:
            spikes = self.hw_observables.get_spikes(self.n_events)
            time = spikes[1] * 1e-6  # Convert to seconds
            perm = np.argsort(time, axis=1)
            time = np.take_along_axis(time, perm, axis=1).astype(np.float32)
            idx = np.take_along_axis(spikes[0].astype(np.int32), perm, axis=1)
            layer_idx = np.where(idx != -1, self.layer_idx, -2).astype(np.int32)
            current = np.zeros_like(idx, dtype=np.float32)
            internal = np.where(idx != -1, True, False)

            return Spike(
                time=time,
                idx=idx,
                current=current,
                layer_idx=layer_idx,
                internal=internal,
            )

        return None
