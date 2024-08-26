from typing import Optional

import jax
import jax.numpy as np
from jaxsnn.event.types import EventPropSpike


def spatio_temporal_encode(
    inputs: jax.Array,
    t_late: float,
    duplication: Optional[int],
    duplicate_neurons: bool,
) -> EventPropSpike:
    spike_idx = np.arange(inputs.shape[-1])

    if duplication is not None:
        inputs = np.repeat(inputs, duplication, axis=-1)
        if duplicate_neurons:
            # duplicate over multiple neurons
            spike_idx = np.arange(spike_idx.shape[0] * duplication)
        else:
            spike_idx = np.repeat(spike_idx, duplication, axis=-1)

    assert spike_idx.shape == inputs.shape

    # sort spikes
    sort_idx = np.argsort(inputs, axis=-1)
    inputs = inputs[sort_idx]
    spike_idx = spike_idx[sort_idx]

    inputs = inputs * t_late

    # Add zero current
    input_spikes = EventPropSpike(
        inputs, spike_idx, np.zeros_like(spike_idx, dtype=inputs.dtype)
    )
    return input_spikes


def target_temporal_encode(
    targets: jax.Array,
    correct_target_time: float,
    wrong_target_time: float,
    n_classes: int
) -> jax.Array:
    encoding = np.full((n_classes, n_classes), wrong_target_time)
    diag_indices = np.diag_indices_from(encoding)
    encoding = encoding.at[diag_indices].set(correct_target_time)

    target_spike_times = encoding[targets]

    return target_spike_times
