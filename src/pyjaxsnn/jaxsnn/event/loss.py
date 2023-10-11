import logging
from typing import Callable, List, Tuple

import jax
import jax.numpy as np
from jaxsnn.base.types import ArrayLike
from jaxsnn.event import custom_lax
from jaxsnn.event.types import (
    Apply,
    ApplyHW,
    EventPropSpike,
    LIFState,
    LossAndRecording,
    TestResult,
    Weight,
)

log = logging.getLogger(__name__)


def max_over_time(output: LIFState) -> jax.Array:
    return np.max(output.V, axis=0)


def nll_loss(output: jax.Array, targets: jax.Array) -> float:
    n_classes = targets.shape[0]
    idx = np.argmin(targets)
    targets = np.array(idx == np.arange(n_classes))
    output = np.maximum(output, 0)
    preds = jax.nn.log_softmax(output)
    loss = -np.sum(targets * preds)
    return loss


def max_over_time_loss(
    apply_fn: Apply,
    weights: List[Weight],
    batch: Tuple[EventPropSpike, jax.Array],
) -> Tuple[ArrayLike, Tuple[float, List[EventPropSpike]]]:
    input_spikes, target = batch
    recording = apply_fn(weights, input_spikes)
    output = recording[-1]
    assert isinstance(output, LIFState)
    max_voltage = max_over_time(output)
    loss_value = nll_loss(max_voltage, target)

    # return negative voltage as accuracy takes the minimum of this value
    return loss_value, (-max_voltage, recording)


def loss_wrapper(  # pylint: disable=too-many-arguments
    apply_fn: Apply,
    loss_fn: Callable[[jax.Array, jax.Array, float], float],
    tau_mem: float,
    n_neurons: int,
    n_outputs: int,
    weights: List[Weight],
    batch: Tuple[EventPropSpike, jax.Array],
) -> LossAndRecording:
    input_spikes, target = batch
    recording = apply_fn(weights, input_spikes)
    output = recording[-1]
    t_first_spike = first_spike(output, n_neurons)[n_neurons - n_outputs:]
    loss_value = loss_fn(t_first_spike, target, tau_mem)

    return loss_value, (t_first_spike, recording)


def loss_wrapper_known_spikes(  # pylint: disable=too-many-arguments
    apply_fn: ApplyHW,
    loss_fn: Callable[[jax.Array, jax.Array, float], float],
    tau_mem: float,
    n_neurons: int,
    n_outputs: int,
    weights: List[Weight],
    batch: Tuple[EventPropSpike, jax.Array],
    spikes: List[EventPropSpike],
) -> LossAndRecording:
    input_spikes, target = batch
    recording = apply_fn(spikes, weights, input_spikes)
    output = recording[-1]
    t_first_spike = first_spike(output, n_neurons)[n_neurons - n_outputs:]
    loss_value = loss_fn(t_first_spike, target, tau_mem)

    return loss_value, (t_first_spike, recording)


def target_time_loss(
    first_spikes: jax.Array, target: jax.Array, tau_mem: float
) -> float:
    loss_value = -np.sum(
        np.log(1 + np.exp(-np.abs(first_spikes - target) / tau_mem))
    )
    return loss_value


def ttfs_loss(
    first_spikes: jax.Array, target: jax.Array, tau_mem: float
) -> float:
    idx = np.argmin(target)
    first_spikes = np.minimum(np.abs(first_spikes), 2 * tau_mem)
    return -np.log(
        np.sum(np.exp((first_spikes[idx] - first_spikes) / tau_mem))
    )


def mse_loss(
    first_spikes: jax.Array, target: jax.Array, tau_mem: float
) -> float:
    return np.sum(
        np.square((np.minimum(first_spikes, 2 * tau_mem) - target) / tau_mem)
    )


def first_spike(spikes: EventPropSpike, size: int) -> jax.Array:
    return np.array(
        [
            np.min(np.where(spikes.idx == idx, spikes.time, np.inf))
            for idx in range(size)
        ]
    )


def loss_and_acc(
    loss_fn: Callable,
    weights: List[Weight],
    dataset: Tuple[EventPropSpike, jax.Array],
) -> TestResult:
    batched_loss = jax.vmap(loss_fn, in_axes=(None, 0))
    loss, (t_first_spike, recording) = batched_loss(weights, dataset)
    accuracy = np.argmin(dataset[1], axis=-1) == np.argmin(
        t_first_spike, axis=-1
    )
    return TestResult(
        np.mean(loss),
        np.mean(accuracy),
        t_first_spike,
        recording,
    )


def loss_and_acc_scan(
    loss_fn: Callable,
    weights: List[Weight],
    dataset: Tuple[EventPropSpike, jax.Array],
) -> TestResult:
    weights, (loss, (t_first_spike, recording)) = custom_lax.scan(
        loss_fn, weights, dataset
    )
    accuracy = np.argmin(dataset[1], axis=-1) == np.argmin(
        t_first_spike, axis=-1
    )
    return TestResult(
        np.mean(loss),
        np.mean(accuracy),
        t_first_spike,
        recording,
    )
