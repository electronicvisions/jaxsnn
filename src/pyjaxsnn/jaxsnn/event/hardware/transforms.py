import jax
import jax.numpy as jnp


def linear_saturating(
    weight: jax.Array,
    scale: float,
    min_weight: float = -63.0,
    max_weight: float = 63.0,
    as_int: bool = True,
) -> jax.Array:
    """
    Scale all weights according to:

        w <- clip(scale * w, min_weight, max_weight)

    :param weight: The weight array to be transformed.
    :param scale: A constant the weight array is scaled with.
    :param min_weight: The minimum value, smaller values are clipped to after
        scaling.
    :param max_weight: The maximum value, bigger values are clipped to after
        scaling.
    :param as_int: Round to nearest int and return as int type.

    :returns: The transformed weight tensor.
    """
    if as_int:
        return jnp.round(
            jnp.clip(scale * weight, min_weight, max_weight)
        ).astype(int)
    return jnp.clip(scale * weight, min_weight, max_weight)
