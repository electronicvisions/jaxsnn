from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from jaxsnn.discrete.types import (
    SourcePopulation,
    Parameter,
    DenseData,
)


def Input(  # pylint: disable=invalid-name
    size: int
) -> SourcePopulation:
    """
    Create an input layer descriptor.

    It returns None for the generator because it's is not needed.

    :param size: Number of neurons in the input layer.

    :returns: A SourcePopulation instance representing the input layer.
    """
    def generator(
        dt: float,  # pylint: disable=unused-argument
    ) -> SourcePopulation.Functions:
        """ """
        def init_fn(
            rng: jax.Array,  # pylint: disable=unused-argument
        ) -> Optional[Parameter]:
            """ """
            return None

        def state_fn() -> Tuple[None, DenseData]:
            """ """
            return None, jnp.zeros(size)

        def step_fn(
            inputs: Dict[str, DenseData],
            state: None,  # pylint: disable=unused-argument
            parameters: Optional[Parameter],  # pylint: disable=unused-argument
        ) -> Tuple[None, DenseData]:
            assert len(inputs) == 1, "Input layer only supports one input"
            return None, list(inputs.values())[0]

        return SourcePopulation.Functions(init_fn, state_fn, step_fn)

    parameters = {
        "size": size,
    }

    return SourcePopulation(generator, parameters, size)
