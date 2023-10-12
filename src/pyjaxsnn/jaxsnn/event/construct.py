from typing import List, Optional, Tuple

import jax
import jax.numpy as np
from jaxsnn.event.types import SingleInit, WeightInput, WeightRecurrent


def construct_recurrent_init_fn(
    layers: List[int],
    mean: List[float],
    std: List[float],
    duplication: Optional[float] = None,
) -> SingleInit:
    def init_fn(
        rng: jax.random.KeyArray, input_size: int
    ) -> Tuple[int, WeightRecurrent]:
        assert len(layers) >= 1
        hidden_size = np.sum(np.array(layers))

        rng, layer_rng = jax.random.split(rng)
        if duplication is not None:
            input_weights = jax.random.normal(
                layer_rng, (int(input_size / duplication), layers[0])
            )
            input_weights = np.repeat(input_weights, duplication, axis=0)
        else:
            input_weights = jax.random.normal(
                layer_rng, (input_size, layers[0])
            )
        input_weights = (
            np.zeros((input_size, hidden_size))
            .at[:, : layers[0]]
            .set(input_weights * std[0] + mean[0])
        )

        recurrent_weights = np.zeros((hidden_size, hidden_size))
        l_sum = 0
        for i, (layer_1, layer_2) in enumerate(zip(layers, layers[1:])):
            rng, layer_rng = jax.random.split(rng)
            ix_middle = l_sum + layer_1

            recurrent_weights = recurrent_weights.at[
                l_sum:ix_middle, ix_middle: ix_middle + layer_2
            ].set(
                jax.random.normal(layer_rng, (layer_1, layer_2)) * std[i + 1]
                + mean[i + 1]
            )
            l_sum += layer_1

        weights = WeightRecurrent(input_weights, recurrent_weights)
        return hidden_size, weights

    return init_fn


def construct_init_fn(
    n_hidden: int, mean: float, std: float, duplication: Optional[int] = None
) -> SingleInit:
    def init_fn(
        rng: jax.random.KeyArray, input_shape: int
    ) -> Tuple[int, WeightInput]:
        if duplication is not None:
            weights = jax.random.normal(
                rng, (int(input_shape / duplication), n_hidden)
            )
            weights = np.repeat(weights, duplication, axis=0)
        else:
            weights = jax.random.normal(rng, (input_shape, n_hidden))
        return n_hidden, WeightInput(weights * std + mean)

    return init_fn
