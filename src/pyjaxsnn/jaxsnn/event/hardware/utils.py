# pylint: disable=invalid-name,logging-not-lazy,logging-fstring-interpolation
import logging
from functools import partial
from typing import List, Optional

import jax
import jax.numpy as np
import numpy as onp
from jax import random
from jaxsnn.event.types import Spike, Weight, WeightInput, WeightRecurrent

log = logging.getLogger(__name__)


def spike_to_grenade_input(spike: Spike, input_neurons: int):
    """Convert jaxsnn spike representation to grenade spike representation

    We represent spikes as tuple of index and time. An instance of
    grenade.InputGenerator expects a python list with shape
    [badge, neuron_idx, spike_time]

    TODO move to hxtorch / grenade
    """
    batch_dim = spike.time.shape[0]
    spike_dim = spike.time.shape[1]

    batch = []
    for batch_idx in range(batch_dim):
        spikes = [[] for _ in range(input_neurons)]
        for i in range(spike_dim):
            idx, time = spike.idx[batch_idx, i], float(
                spike.time[batch_idx, i]
            )
            spikes[idx].append(time)
        batch.append(spikes)
    return batch


def linear_saturating(
    weight: np.ndarray,
    scale: float,
    min_weight: float = -63.0,
    max_weight: float = 63.0,
    as_int: bool = True,
) -> np.ndarray:
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
        return np.round(
            np.clip(scale * weight, min_weight, max_weight)
        ).astype(int)
    return np.clip(scale * weight, min_weight, max_weight)


def filter_spikes_batch(
    spikes: Spike, layer_start: int, layer_end: Optional[int] = None
):
    """Only return spikes of neurons after layer start

    Other spikes are encoded with time=np.inf and index=-1
    """
    filtered_time = np.where(spikes.idx >= layer_start, spikes.time, np.inf)
    filtered_idx = np.where(spikes.idx >= layer_start, spikes.idx, -1)

    if layer_end is not None:
        filtered_time = np.where(
            filtered_idx < layer_end, filtered_time, np.inf
        )
        filtered_idx = np.where(filtered_idx < layer_end, filtered_idx, -1)

    return sort_batch(Spike(filtered_time, filtered_idx))


def filter_spikes(
    spikes: Spike, layer_start: int, layer_end: Optional[int] = None
):
    """Only return spikes of neurons after layer start

    Other spikes are encoded with time=np.inf and index=-1
    """
    filtered_time = np.where(spikes.idx >= layer_start, spikes.time, np.inf)
    filtered_idx = np.where(spikes.idx >= layer_start, spikes.idx, -1)

    if layer_end is not None:
        filtered_time = np.where(
            filtered_idx < layer_end, filtered_time, np.inf
        )
        filtered_idx = np.where(filtered_idx < layer_end, filtered_idx, -1)

    sort_idx = np.argsort(filtered_time, axis=-1)

    idx = filtered_idx[sort_idx]
    time = filtered_time[sort_idx]
    return Spike(time, idx)


def cut_spikes(spikes: Spike, count):
    return Spike(spikes.time[:count], spikes.idx[:count])


def cut_spikes_batch(spikes: Spike, count):
    return Spike(spikes.time[:, :count], spikes.idx[:, :count])


# sort spikes
def sort_batch(spikes: Spike) -> Spike:
    sort_idx = np.argsort(spikes.time, axis=-1)
    n_spikes = spikes.time.shape[0]
    time = spikes.time[np.arange(n_spikes)[:, None], sort_idx]
    idx = spikes.idx[np.arange(n_spikes)[:, None], sort_idx]
    return Spike(time=time, idx=idx)


def add_noise_batch(
    spikes: Spike, rng: random.PRNGKey, std: float = 1e-7, bias: float = 1e-7
) -> Spike:
    noise = random.normal(rng, spikes.time.shape) * std + bias
    spikes_with_noise = Spike(time=spikes.time + noise, idx=spikes.idx)
    # return spikes_with_noise
    return sort_batch(spikes_with_noise)


def simulate_hw_weights(
    weights: List[Weight], scale: float, as_int: bool = False
) -> List[Weight]:
    new_weights = []
    for weight in weights:
        if isinstance(weight, WeightInput):
            new_weight = WeightInput(
                linear_saturating(weight.input, scale, as_int=as_int) / scale
            )
        else:
            new_weight = WeightRecurrent(
                input=linear_saturating(weight.input, scale, as_int=as_int)
                / scale,
                recurrent=linear_saturating(
                    weight.recurrent, scale, as_int=as_int
                )
                / scale,
            )
        new_weights.append(new_weight)
    return new_weights


def simulate_madc(
    tau_mem_inv: float,
    tau_syn_inv: float,
    inputs: Spike,
    weight: float,
    ts: jax.Array,
):
    A = np.array([[-tau_mem_inv, tau_mem_inv], [0, -tau_syn_inv]])
    tk = inputs.time
    xk = np.array([[0.0, weight]])

    def heaviside(x):
        return 0.5 + 0.5 * np.sign(x)

    def kernel(t, t0):
        return heaviside(t - t0) * jax.scipy.linalg.expm(A * (t - t0))

    def f(t0, x0, t):
        return np.einsum("ijk, ik -> j", jax.vmap(partial(kernel, t))(t0), x0)

    return jax.vmap(partial(f, tk, xk), in_axes=0)(ts)


def add_linear_noise(spike: Spike) -> Spike:
    batch_size, n_spikes = spike.idx.shape
    time_noise = np.repeat(
        np.expand_dims(np.linspace(0, 1e-9, n_spikes), axis=0),
        batch_size,
        axis=0,
    )
    assert time_noise.shape == spike.idx.shape
    return Spike(idx=spike.idx, time=spike.time + time_noise)


def first_spike(spikes: Spike, start: int, stop: int) -> jax.Array:
    return np.array(
        [
            np.min(np.where(spikes.idx == idx, spikes.time, np.inf))
            for idx in range(start, stop)
        ]
    )


first_spike_batch = jax.vmap(first_spike, in_axes=(0, None, None))


def spike_similarity_batch(spike1: Spike, spike2: Spike):
    assert spike1.time.shape == spike2.time.shape

    for start, stop in ((125, 128),):
        first_spike1 = first_spike_batch(spike1, start, stop)
        first_spike2 = first_spike_batch(spike2, start, stop)

        # # now compare the times
        diff = first_spike1 - first_spike2
        masked = onp.ma.masked_where(diff == np.inf, diff)
        masked = onp.ma.masked_where(masked == np.nan, masked)
        log.info(
            f"SW: {np.mean(first_spike1, axis=0) / 6e-6} tau_syn,"
            f"HW: {np.mean(first_spike2, axis=0) / 6e-6} tau_syn"
        )
