from typing import (
    Optional,
    Tuple,
)

import jax
import jax.numpy as jnp


from jaxsnn.discrete.types import (
    Projection,
    Parameter,
    DenseData,
)
from jaxsnn.discrete.functional.linear import linear


def Linear(  # pylint: disable=invalid-name
    mean: float = 0.5,
    std: float = 2.0,
    pre_weights: Optional[jnp.ndarray] = None
) -> Projection:
    """
    Creates a linear projection layer

    Either:
        - initialize weights from a Gaussian (mean, std), or
        - provide a concrete weight array.

    :param mean: Mean value for weight initialization.
    :param std: Standard deviation for weight initialization.
    :param pre_weights: Optional weight array. If provided, mean and std are
        ignored.

    :return: A Projection object containing the layer definition.
    """

    def generator(
        input_size: int,
        output_size: int,
    ) -> Projection.Functions:
        """
        Generates the runtime functions for the linear projection.

        :param input_size: Size of the input layer.
        :param output_size: Size of the output layer.

        :return: A Projection.Functions object holding the collection of
            functions (init, state, step).
        """

        def init_fn(
            rng: jax.Array,
        ) -> Parameter:
            if pre_weights is not None:
                return jnp.asarray(pre_weights)
            weights = jax.random.normal(rng, (input_size, output_size))
            return weights * std + mean

        def state_fn() -> Tuple[None, DenseData]:
            return None, jnp.zeros(output_size)

        return Projection.Functions(init_fn, state_fn, linear)

    parameters = {
        "mean": mean,
        "std": std,
    }

    return Projection(generator, parameters)
