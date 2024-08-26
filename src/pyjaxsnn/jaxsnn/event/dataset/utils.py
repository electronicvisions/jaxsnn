from typing import Tuple, Optional, Any

import jax
from jax import random
import jax.numpy as np
from jaxsnn.event.types import EventPropSpike

Dataset = Tuple[EventPropSpike, jax.Array, str]


def data_loader(
    dataset: Any,
    batch_size: int,
    rng: Optional[random.KeyArray] = None,
):
    total_length = jax.tree_util.tree_leaves(dataset[0])[0].shape[0]

    permutation = (
        random.permutation(rng, total_length)
        if rng is not None
        else np.arange(total_length)
    )
    # Perform permutation of dataset
    dataset = jax.tree_map(lambda x: x[permutation], dataset)

    # Determine number of batches
    num_batches = total_length // batch_size

    # Split dataset up into batches
    dataset = jax.tree_map(
        lambda x: x.reshape((num_batches, batch_size) + x.shape[1:]), dataset
    )

    return dataset
