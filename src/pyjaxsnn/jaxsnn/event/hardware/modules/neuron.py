""" Implementing SNN modules """
from __future__ import annotations
from typing import Optional

import numpy as np

from jaxsnn.event.hardware.observables import Observables
from jaxsnn.event.hardware.modules.population import Population
from jaxsnn.event.types import Spike


class Neuron(Population):

    def __init__(
        self,
        layer_idx: int,
        n_events: int,
        n_hw_spikes: Optional[int] = None,
        *args,
        time_offset: float = 0.0,
        **kwargs
    ) -> None:
        super().__init__(
            layer_idx=layer_idx,
            n_events=n_events,
            n_hw_spikes=n_hw_spikes,
            time_offset=time_offset,
            *args,
            **kwargs,
        )
        self.hw_observables: Observables

    def get_post_processed(self) -> Spike:
        """ """
        spikes = self.hw_observables.get_spikes(
            self.n_events,
            self.n_hw_spikes,
        )

        time = np.maximum((spikes[1] + self.time_offset * 1E6), 0) * 1E-6
        perm = np.argsort(time, axis=1)
        time = np.take_along_axis(
            time,
            perm,
            axis=1,
        ).astype(np.float32)
        idx = np.take_along_axis(
            spikes[0],
            perm,
            axis=1,
        ).astype(np.int32)
        layer_idx = np.where(
            idx != -1,
            self.layer_idx,
            -1
        ).astype(np.int32)
        current = np.zeros_like(idx, dtype=np.float32)
        internal = np.where(idx != -1, True, False)

        spikes = Spike(
            time=time,
            idx=idx,
            current=current,
            layer_idx=layer_idx,
            internal=internal,
        )

        return spikes
