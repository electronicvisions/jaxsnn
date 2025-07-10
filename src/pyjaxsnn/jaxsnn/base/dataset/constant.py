from typing import Tuple

import jax
import jax.numpy as jnp


def constant_dataset(t_max: float, size: int) -> Tuple[jax.Array, jax.Array]:
    inputs = jnp.array([0.1, 0.2, 1]) * t_max
    target = jnp.array([0.2, 0.3]) * t_max

    # Duplicate input
    inputs = jnp.tile(inputs, (size, 1))
    target = jnp.tile(target, (size, 1))

    return inputs, target, "constant"
