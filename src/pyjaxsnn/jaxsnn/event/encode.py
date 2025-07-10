from typing import Optional, Tuple, Callable

import jax
import jax.numpy as jnp
from jaxsnn.event.types import EventPropSpike


def spatio_temporal_encode(
    inputs: jax.Array,
    t_late: float,
    duplication: Optional[int],
    duplicate_neurons: bool,
) -> EventPropSpike:
    spike_idx = jnp.arange(inputs.shape[-1])

    if duplication is not None:
        inputs = jnp.repeat(inputs, duplication, axis=-1)
        if duplicate_neurons:
            # duplicate over multiple neurons
            spike_idx = jnp.arange(spike_idx.shape[0] * duplication)
        else:
            spike_idx = jnp.repeat(spike_idx, duplication, axis=-1)

    assert spike_idx.shape == inputs.shape

    # sort spikes
    sort_idx = jnp.argsort(inputs, axis=-1)
    inputs = inputs[sort_idx]
    spike_idx = spike_idx[sort_idx]

    inputs = inputs * t_late

    # Add zero current
    input_spikes = EventPropSpike(
        inputs, spike_idx, jnp.zeros_like(spike_idx, dtype=inputs.dtype)
    )
    return input_spikes


def target_temporal_encode(
    targets: jax.Array,
    correct_target_time: float,
    wrong_target_time: float,
    n_classes: int
) -> jax.Array:
    encoding = jnp.full((n_classes, n_classes), wrong_target_time)
    diag_indices = jnp.diag_indices_from(encoding)
    encoding = encoding.at[diag_indices].set(correct_target_time)

    target_spike_times = encoding[targets]

    return target_spike_times


def target_one_hot_encode(
    target: jax.Array,
    scale: float,
    n_classes: int
) -> jax.Array:
    encoding = jnp.zeros((n_classes))
    encoding = encoding.at[target].set(scale)
    return encoding


def encode(
    dataset: Tuple[jax.Array, jax.Array],
    input_encoder: Optional[Callable] = None,
    target_encoder: Optional[Callable] = None
):
    inputs = dataset[0]
    targets = dataset[1]
    if input_encoder is not None:
        inputs = input_encoder(inputs)

    if target_encoder is not None:
        targets = target_encoder(targets)

    return (inputs, targets)
