from typing import Callable, Optional

import jax

from pyhalco_hicann_dls_vx_v3 import DLSGlobal

from jaxsnn.base.types import Parameter
from jaxsnn.event.types import Projection
from jaxsnn.event.hardware.transforms import linear_saturating
from jaxsnn.event.hardware.experiment import Experiment
from jaxsnn.event.hardware.modules.population import Population
from jaxsnn.event.hardware.modules.synapse import Synapse


def HXLinear(  # pylint: disable=invalid-name, too-many-arguments
    mean: float = 0.5,
    std: float = 2.0,
    min_delay: float = 0.0,
    weight_scale: float = 1.0,
    chip_coordinate: Optional[DLSGlobal] = None,
    transform: Callable = linear_saturating,
) -> Projection:
    """
    Creates a synapse initialization function and associated parameters.

    :param mean: Mean value for initializing synaptic weights.
    :param std: Standard deviation for initializing synaptic weights.
    :param min_delay: Minimum allowable synaptic delay.
    :param weight_scale: Scaling factor for synaptic weights.
    :param chip_coordinate: Chip coordinate for hardware execution.
    :param transform: Transformation function applied to synaptic weights.

    :returns: Projection instance containing generator function and parameters.
    """

    def generator(
        input_size: int,
        output_size: int,
    ) -> Projection.Functions:

        def init_fn(
            rng: jax.Array,
        ) -> Parameter:
            weights = jax.random.normal(rng, (input_size, output_size))
            return weights * std + mean

        def state_fn(
            *args,  # pylint: disable=unused-argument
        ) -> None:
            return None

        def event_fn(
            *args,  # pylint: disable=unused-argument
        ) -> None:
            return None

        def generate_hx_module_fn(
            layer_idx: int,
            experiment: Experiment,
            source: Population,
            target: Population,
        ) -> Synapse:
            return Synapse(
                layer_idx=layer_idx,
                source_population=source,
                target_population=target,
                experiment=experiment,
                chip_coordinate=chip_coordinate,
                weight_scale=weight_scale,
                transform=transform,
            )
        return Projection.Functions(
            init_fn,
            state_fn,
            event_fn,
            generate_hx_module_fn,
        )

    parameters = {
        "mean": mean,
        "std": std,
        "min_delay": min_delay,
        "weight_scale": weight_scale,
        "chip_coordinate": chip_coordinate,
        "transform": transform,
    }

    return Projection(generator, parameters, min_delay)
