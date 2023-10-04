from typing import Callable, List, Tuple, Union

import jax
import jax.numpy as np
from jaxsnn.event import custom_lax
from jaxsnn.base.types import Array, ArrayLike, Spike, Weight
from jaxsnn.functional.leaky_integrate_and_fire import LIFState
import logging


log = logging.getLogger(__name__)


def max_over_time(output: LIFState) -> Array:
    return np.max(output.V, axis=0)


def nll_loss(x: Array, targets: Array) -> ArrayLike:
    n_classes = targets.shape[0]
    idx = np.argmin(targets)
    targets = np.array(idx == np.arange(n_classes))
    x = np.maximum(x, 0)
    preds = jax.nn.log_softmax(x)
    loss = -np.sum(targets * preds)
    return loss


def max_over_time_loss(
    apply_fn: Callable[[List[Weight], Spike], List[Union[Spike, LIFState]]],
    weights: List[Weight],
    batch: Tuple[Spike, Array],
) -> Tuple[ArrayLike, Tuple[ArrayLike, List[Union[Spike, LIFState]]]]:
    input_spikes, target = batch
    recording = apply_fn(weights, input_spikes)
    output = recording[-1]
    assert isinstance(output, LIFState)
    max_voltage = max_over_time(output)
    loss_value = nll_loss(max_voltage, target)

    # return negative voltage as accuracy takes the minimum of this value
    return loss_value, (-max_voltage, recording)


def loss_wrapper(
    apply_fn: Callable[[List[Weight], Spike], List[Spike]],
    loss_fn: Callable[[Array, Array, float], ArrayLike],
    tau_mem: float,
    n_neurons: int,
    n_outputs: int,
    weights: List[Weight],
    batch: Tuple[Spike, Array],
) -> Tuple[ArrayLike, Tuple[ArrayLike, List[Spike]]]:
    input_spikes, target = batch
    recording = apply_fn(weights, input_spikes)
    output = recording[-1]
    t_first_spike = first_spike(output, n_neurons)[n_neurons - n_outputs :]
    loss_value = loss_fn(t_first_spike, target, tau_mem)

    return loss_value, (t_first_spike, recording)


def loss_wrapper_known_spikes(
    apply_fn: Callable[[List[Spike], List[Weight], Spike], List[Spike]],
    loss_fn: Callable[[Array, Array, float], ArrayLike],
    tau_mem: float,
    n_neurons: int,
    n_outputs: int,
    weights: List[Weight],
    batch: Tuple[Spike, Array],
    spikes: List[Spike],
) -> Tuple[ArrayLike, Tuple[ArrayLike, List[Spike]]]:
    input_spikes, target = batch
    recording = apply_fn(spikes, weights, input_spikes)
    output = recording[-1]
    t_first_spike = first_spike(output, n_neurons)[n_neurons - n_outputs :]
    loss_value = loss_fn(t_first_spike, target, tau_mem)

    return loss_value, (t_first_spike, recording)


def target_time_loss(first_spikes: Array, target: Array, tau_mem: float) -> ArrayLike:
    loss_value = -np.sum(np.log(1 + np.exp(-np.abs(first_spikes - target) / tau_mem)))
    return loss_value


def adapted_ttfs_loss(first_spikes: Array, target: Array, tau_mem: float) -> ArrayLike:
    idx = np.argmin(target)
    return -np.log(
        np.sum(
            1 + np.exp(-first_spikes[idx] / tau_mem) - np.exp(-first_spikes / tau_mem)
        )
    )


def ttfs_loss(first_spikes: Array, target: Array, tau_mem: float) -> ArrayLike:
    idx = np.argmin(target)
    first_spikes = np.minimum(np.abs(first_spikes), 2 * tau_mem)
    return -np.log(np.sum(np.exp((first_spikes[idx] - first_spikes) / tau_mem)))


def mse_loss(first_spikes: Array, target: Array, tau_mem: float) -> ArrayLike:
    return np.sum(np.square((np.minimum(first_spikes, 2 * tau_mem) - target) / tau_mem))


def adapted_event_prop_loss(
    first_spikes: Array, target: Array, tau_mem: float
) -> ArrayLike:
    idx = np.argmin(target)
    numerator = np.exp(-first_spikes[idx] / tau_mem)
    denominator = np.exp(-np.abs(first_spikes) / tau_mem)

    alpha = 0.1
    regularization = alpha * np.square(first_spikes[idx] / tau_mem - 0.5)

    loss_value = -np.log(1 + numerator / (1 + np.sum(denominator))) + regularization
    return loss_value


def event_prop_loss(first_spikes: Array, target: Array, tau_mem: float) -> ArrayLike:
    # maybe also needs regularization ?
    idx = np.argmin(target)
    first_spikes = np.minimum(np.abs(first_spikes), 2 * tau_mem)
    numerator = np.exp(-first_spikes[idx] / tau_mem)
    denominator = np.exp(-first_spikes / tau_mem)

    loss_value = -np.log(numerator / np.sum(denominator))

    alpha = 0.5
    regularization = alpha * (np.exp(first_spikes[idx] / tau_mem) - 1)
    return loss_value + regularization


def first_spike(spikes: Spike, size: int) -> Array:
    return np.array(
        [
            np.min(np.where(spikes.idx == idx, spikes.time, np.inf))
            for idx in range(size)
        ]
    )


def loss_and_acc(
    loss_fn: Callable,
    params: List[Weight],
    dataset: Tuple[Spike, Array],
):
    batched_loss = jax.vmap(loss_fn, in_axes=(None, 0))
    loss, (t_first_spike, recording) = batched_loss(params, dataset)
    accuracy = np.argmin(dataset[1], axis=-1) == np.argmin(t_first_spike, axis=-1)
    return (
        np.mean(loss),
        np.mean(accuracy),
        t_first_spike,
        recording,
    )


def loss_and_acc_scan(loss_fn, params, dataset):
    params, (loss, (t_first_spike, recording)) = custom_lax.simple_scan(
        loss_fn, params, dataset
    )
    accuracy = np.argmin(dataset[1], axis=-1) == np.argmin(t_first_spike, axis=-1)
    return (
        np.mean(loss),
        np.mean(accuracy),
        t_first_spike,
        recording,
    )
