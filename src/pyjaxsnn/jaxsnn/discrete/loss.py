import jax
import jax.numpy as jnp
from jaxsnn.discrete.encode import one_hot


def nll_loss(predictions: jax.Array, targets: jax.Array) -> jax.Array:
    targets = one_hot(targets, predictions.shape[1])
    loss = -jnp.mean(jnp.sum(targets * predictions, axis=1))
    return loss
