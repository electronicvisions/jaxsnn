from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Optional,
)
import time

from jaxsnn import get_logger
from hxtorch.core.experiment import BaseExperiment

if TYPE_CHECKING:
    from jaxsnn.event.topology import Topology
    from jaxsnn.event.types import (
        Parameters,
        IOData,
    )


class Experiment(BaseExperiment):
    """
    Hardware experiment class for executing experiment on BrainScaleS-2
    """

    def __init__(
        self,
        topology: Topology,
        *args,
        inter_batch_entry_wait: int = 0,
        **kwargs
    ) -> None:
        """
        Initialize the hardware experiment.

        :param topology: The network topology defining the structure and
            connections.
        :param inter_batch_entry_wait: Wait time between batch entries in
            FPGA cycles.
        :param args: Additional positional arguments passed to BaseExperiment.
        :param kwargs: Additional keyword arguments passed to BaseExperiment.
        """
        self.topology = topology
        super().__init__(*args, inter_batch_entry_wait, **kwargs)
        self.log = get_logger("jaxsnn.event.hardware.Experiment")
        self.runtime_in_s: float = 0.0
        self._batch_size: Optional[int] = None

    def expected_return_type(
        self,
        input_spikes: IOData
    ) -> IOData:
        self.batch_size = input_spikes
        return_type: IOData = {}
        for node, attrs in self.topology.graph.nodes(data=True):
            hx_module = attrs["hx_module"]
            return_type[node] = hx_module.expected_return_type
        self.log.TRACE(f"Expected return type: {return_type}")

        return return_type

    @property
    def batch_size(self) -> int:
        """
        Returns the batch size of the experiment.
        """
        if self._batch_size is None:
            raise ValueError(
                "Batch size is not set. Set batch_size first.",
            )
        return self._batch_size

    @batch_size.setter
    def batch_size(
        self,
        input_spikes: Optional[IOData],
    ) -> None:
        """
        Sets the batch size of the experiment.
        """
        if input_spikes is None:
            self._batch_size = None
            return

        # Check batch size
        batch_sizes = [
            spikes.time.shape[0] for spikes in input_spikes.values()
        ]
        if not all(bs == batch_sizes[0] for bs in batch_sizes):
            raise ValueError(
                "All inputs must have the same batch size. "
                f"Found sizes: {batch_sizes}"
            )
        self._batch_size = batch_sizes[0]
        self.log.TRACE(f"Batch size: {self._batch_size}")

    def run(  # pylint: disable=arguments-differ
        self,
        input_spikes: IOData,
        parameters: Parameters,
    ) -> IOData:
        """
        Executes the experiment in mock-mode or on hardware using the
        information added to the experiment for a time given by `runtime` and
        returns a dict of hardware data represented as PyTorch data types.

        :param runtime: The runtime of the experiment on hardware in µs.

        :returns: Returns the data map as dict, where the keys are the
            population descriptors and values are tuples of values returned by
            the corresponding module's `post_process` method.
        """
        # Assign inputs and parameters to modules
        for node, attrs in self.topology.graph.nodes(data=True):
            hx_module = attrs["hx_module"]
            if hasattr(hx_module, "set_spike_times"):
                hx_module.set_spike_times(input_spikes[node])
            if hasattr(hx_module, "set_params"):
                hx_module.set_params(parameters[node])

        super().run(self.runtime_in_s)

        tic = time.perf_counter()
        hx_spikes: IOData = {}
        for node, attrs in self.topology.graph.nodes(data=True):
            hx_module = attrs["hx_module"]
            hx_spikes[node] = hx_module.get_post_processed()
        for key, spikes in input_spikes.items():
            if hx_spikes[key] is None:
                hx_spikes[key] = spikes

        toc = time.perf_counter()
        self.log.TRACE(
            f"Post-processing spike merging runtime: {(toc - tic):.4f} "
            "seconds."
        )

        return hx_spikes
