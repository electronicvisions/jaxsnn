from typing import (
    Tuple,
    Dict,
)

import jax.numpy as jnp

from jaxsnn.discrete.types import (
    Parameter,
    DenseData,
)


def linear(
    inputs: Dict[str, DenseData],
    state: None,  # pylint: disable=unused-argument
    weight: Parameter,
) -> Tuple[None, DenseData]:
    assert len(inputs) == 1, "Linear layer only supports one input"
    return None, jnp.matmul(list(inputs.values())[0], weight)
