from typing import Optional

import jax
import jax.numpy as jnp

from jaxsnn.base.types import Parameter
from jaxsnn.event.types import Projection


# pylint: disable=invalid-name
def Linear(
    mean: float = 0.5,
    std: float = 2.0,
    min_delay: float = 0.0,
    pre_weights: Optional[jnp.ndarray] = None
) -> Projection:
    """
    Creates a Linear projection layer

    Either:
        - initialize weights from a Gaussian (mean, std), or
        - provide a concrete weight array.

    :param mean: Mean of the Gaussian distribution for weight initialization.
    :param std: Standard deviation of the Gaussian distribution.
    :param min_delay: Minimum delay associated with this projection.
    :param pre_weights: Optional weight array. If provided, mean and std are
        ignored.

    :returns: A Projection object containing the generator and parameters.
    """

    # pylint: disable=unused-argument
    def generator(
        input_size: int,
        output_size: int,
    ) -> Projection.Functions:
        """
        Generates the initialization and state functions for the projection.

        :param input_size: Size of the input layer.
        :param output_size: Size of the output layer.

        :returns: A Projection.Functions object containing init, state, and
            event functions.
        """
        def init_fn(rng: jax.Array) -> Parameter:
            if pre_weights is not None:
                return jnp.asarray(pre_weights)
            weights = jax.random.normal(rng, (input_size, output_size))
            return weights * std + mean

        def state_fn(*args) -> None:
            return None

        def event_fn(*args) -> None:
            return None

        return Projection.Functions(init_fn, state_fn, event_fn)

    parameters = {
        "mean": mean,
        "std": std,
        "min_delay": min_delay
    }

    return Projection(generator, parameters, min_delay)
