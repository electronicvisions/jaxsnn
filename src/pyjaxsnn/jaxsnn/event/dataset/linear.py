from typing import Optional, Tuple

import jax
import jax.numpy as np
from jax import random


def linear_dataset(
    rng: random.KeyArray,
    size: int,
    mirror: bool,
    bias_spike: Optional[float],
) -> Tuple[jax.Array, jax.Array]:

    coords = random.uniform(rng, (size, 2))
    classes = (coords[:, 0] < coords[:, 1]).astype(int)

    if mirror:
        coords = np.hstack((coords, 1 - coords))

    if bias_spike is not None:
        bias = np.full(size, bias_spike)[:, None]
        coords = np.hstack((coords, bias))

    return coords, classes, "constant"
