from typing import Tuple, Optional, Any

import jax
from jax import random
import jax.numpy as jnp
from jaxsnn.event.types import EventPropSpike

Dataset = Tuple[EventPropSpike, jax.Array, str]


def data_loader(
    dataset: Any,
    batch_size: int,
    num_batches: Optional[int] = None,
    rng: Optional[jax.Array] = None,
):
    # Determine number of batches
    if num_batches is None:
        num_batches_list = jax.vmap(
            lambda data: data.shape[0] // batch_size
        )(jnp.array(jax.tree_util.tree_leaves(dataset[0])))
        assert jnp.all(num_batches_list == num_batches_list[0]), \
            "All inputs must have equal size"
        num_batches = num_batches_list[0]

    if rng is not None:
        permutation = random.permutation(rng, num_batches * batch_size)
    else:
        permutation = jnp.arange(num_batches * batch_size)

    # Perform permutation of dataset
    dataset = jax.tree_map(lambda x: jnp.take(x, permutation, axis=0), dataset)

    # Function to split dataset into batches
    def batch_fn(i, data):
        return jax.tree_map(
            lambda x: jax.lax.dynamic_slice_in_dim(
                x, i * batch_size, batch_size), data)

    batched_dataset = jax.vmap(
        batch_fn, in_axes=(0, None))(jnp.arange(num_batches), dataset)

    return batched_dataset
