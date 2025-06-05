# pylint: disable=logging-not-lazy,logging-fstring-interpolation
""" Implementing SNN modules """
from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Optional,
    Tuple,
    List,
    Callable,
)

import numpy as np
import jax

from hxtorch.core.modules.projection import ProjectionConnection
from jaxsnn.event.hardware.modules.projection import Projection

if TYPE_CHECKING:
    from jaxsnn.event.hardware.modules.population import Population


class Synapse(Projection):
    """ Synapse layer """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        layer_idx: int,
        source_population: Population,
        target_population: Population,
        *args,
        weight_scale: float,
        transform: Callable[[jax.Array, float], jax.Array],
        **kwargs,
    ) -> None:
        """
        TODO: Think about what to do with device here.

        :param in_features: Size of input dimension.
        :param out_features: Size of output dimension.
        :param device: Device to execute on. Only considered in mock-mode.
        :param dtype: Data type of weight tensor.
        :param experiment: Experiment to append layer to.
        :param execution_instance: Execution instance to place to.
        """
        super().__init__(
            layer_idx,
            source_population.size,
            target_population.size,
            *args,
            **kwargs,
        )
        self.weight_scale = weight_scale
        self.weight: Optional[jax.Array] = None
        self._weight_hash: Optional[int] = None
        self.weight_transform = transform

        self._source_population: Population = source_population
        self._target_population: Population = target_population

    def source_population(self):
        return self._source_population

    def target_population(self):
        return self._target_population

    @property
    def changed_input_data(self) -> bool:
        """
        Getter for changed_input_data.

        :returns: Boolean indicating whether module changed since last run.
        """
        if self._weight_hash is None:
            return True
        return not hash(self.weight.tobytes()) == self._weight_hash

    @changed_input_data.setter
    def changed_input_data(self, changed: bool) -> None:
        if hasattr(self, "weight") and self.weight is not None:
            self._weight_hash = hash(self.weight.tobytes())

    def get_connections(self) -> List[Tuple[int, int, int]]:
        assert self.weight is not None
        weight_transformed = self.weight_transform(
            self.weight.copy(),
            self.weight_scale,
        )
        connections = [
            ProjectionConnection(col, row, int(weight))
            for (row, col), weight in np.ndenumerate(
                weight_transformed.T.round().astype(int)
            )
        ]
        return connections
