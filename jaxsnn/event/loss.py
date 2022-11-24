from typing import Callable, List, Tuple
import jax.numpy as np

from jaxsnn.base.types import Array, Spike, Weight


def spike_time_loss(
    apply_fn: Callable[[Weight, Spike], List[Spike]],
    tau_mem: float,
    weights: List[Weight],
    batch: Tuple[Spike, Array],
) -> Tuple[float, Tuple[float, List[Spike]]]:

    input_spikes, target = batch
    recording = apply_fn(weights, input_spikes)
    output = recording[-1]
    size = weights[-1].shape[1]  # type: ignore
    t_first_spike = first_spike(output, size)

    return (log_loss(t_first_spike, target, tau_mem), (t_first_spike, recording))


def log_loss(first_spikes: Array, target: Array, tau_mem: float):
    loss_value = -np.sum(np.log(1 + np.exp(-np.abs(first_spikes - target) / tau_mem)))
    return loss_value


def log_loss_single(first_spikes: Array, target: Array, tau_mem: float):
    """Only consider spike time of correct class neuron"""
    idx = np.argmin(target)
    first_spikes = first_spikes[idx]
    target = target[idx]
    loss_value = -np.sum(np.log(1 + np.exp(-np.abs(first_spikes - target) / tau_mem)))
    return loss_value


def first_spike(spikes: Spike, size: int) -> Array:
    return np.array(
        [
            np.min(np.where(spikes.idx == idx, spikes.time, np.inf))
            for idx in range(size)
        ]
    )
