"""
Implementing SNN modules
"""
from typing import Tuple
import jax.numpy as np
import numpy as onp
from jaxsnn.base.types import Array
from jaxsnn.event.hardware import utils

import pygrenade_vx.network.placed_logical as grenade

from jaxsnn.event.hardware.module import Module
import hxtorch
import _hxtorch_core

log = hxtorch.logger.get("hxtorch.snn.modules")


class Synapse(Module):
    """
    Synapse layer
    """

    __constants__ = ["in_features", "out_features"]
    weight: Array

    def __init__(self, experiment, weight: Array) -> None:
        """
        :param in_features: Size of input dimension.
        :param out_features: Size of output dimension.
        :param experiment: Experiment to append layer to.
        """
        super().__init__(experiment=experiment)
        self.weight = weight
        self.register_hw_entity()

    def register_hw_entity(self) -> None:
        """
        Add the synapse layer to the experiment's projections.
        """
        self.experiment.register_projection(self)

    def add_to_network_graph(
        self,
        builder: grenade.NetworkBuilder,
        pre: grenade.PopulationDescriptor,
        post: grenade.PopulationDescriptor,
        scale: float,
    ) -> Tuple[grenade.ProjectionDescriptor, ...]:
        """
        Adds the projection to a grenade network builder by providing the
        population descriptor of the corresponding pre and post population.
        Note: This creates one inhibitory and one excitatory population on
        hardware in order to represent signed hardware weights.

        :param builder: Greande netowrk builder to add projection to.
        :param pre: Population descriptor of pre-population.
        :param post: Population descriptor of post-population.

        :returns: A tuple of grenade ProjectionDescriptors holding the
            descriptors for the excitatory and inhibitory projection.
        """
        weight_exc = np.copy(self.weight)
        weight_inh = np.copy(self.weight)

        weight_exc = utils.linear_saturating(weight_exc, scale=scale, min_weight=0.0)
        weight_inh = utils.linear_saturating(weight_inh, scale=scale, max_weight=0.0)

        connections_exc = _hxtorch_core.weight_to_connection(onp.array(weight_exc))
        connections_inh = _hxtorch_core.weight_to_connection(onp.array(weight_inh))

        projection_exc = grenade.Projection(
            grenade.Receptor(grenade.Receptor.ID(), grenade.Receptor.Type.excitatory),
            connections_exc,
            pre,
            post,
        )
        projection_inh = grenade.Projection(
            grenade.Receptor(grenade.Receptor.ID(), grenade.Receptor.Type.inhibitory),
            connections_inh,
            pre,
            post,
        )

        exc_descriptor = builder.add(projection_exc)
        inh_descriptor = builder.add(projection_inh)
        self.descriptor = (exc_descriptor, inh_descriptor)
        log.TRACE(f"Added projection '{self}' to grenade graph.")

        return self.descriptor
