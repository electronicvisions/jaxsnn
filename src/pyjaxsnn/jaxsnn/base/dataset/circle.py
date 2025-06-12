from typing import Optional, Tuple

import jax
import jax.numpy as np
from jax import random


def circle_dataset(
    rng: jax.Array,
    size: int,
    mirror: bool = True,
    bias_spike: Optional[float] = 0.0,
) -> Tuple[jax.Array, jax.Array]:
    coords = random.uniform(rng, (size, 2))

    # determine class
    center = (0.5, 0.5)
    radius = np.sqrt(0.5 / np.pi)  # spread classes equal
    classes = (
        ((coords[:, 0] - center[0]) ** 2 + (coords[:, 1] - center[1]) ** 2)
        <= radius**2
    ).astype(int)

    if mirror:
        coords = np.hstack((coords, 1 - coords))

    if bias_spike is not None:
        bias = np.full(size, bias_spike)[:, None]
        coords = np.hstack((coords, bias))

    return coords, classes, "circle"
