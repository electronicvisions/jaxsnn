from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import random


def linear_dataset(
    rng: jax.Array,
    size: int,
    mirror: bool,
    bias_spike: Optional[float],
) -> Tuple[jax.Array, jax.Array]:

    coords = random.uniform(rng, (size, 2))
    classes = (coords[:, 0] < coords[:, 1]).astype(int)

    if mirror:
        coords = jnp.hstack((coords, 1 - coords))

    if bias_spike is not None:
        bias = jnp.full(size, bias_spike)[:, None]
        coords = jnp.hstack((coords, bias))

    return coords, classes, "constant"
