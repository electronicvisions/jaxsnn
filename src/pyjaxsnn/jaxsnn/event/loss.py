from typing import Callable, List, Tuple, Any, Optional

import jax
import jax.numpy as jnp
from jaxsnn.base.types import ArrayLike
from jaxsnn.event import custom_lax
from jaxsnn.event.types import (
    Apply,
    EventPropSpike,
    LIFState,
    LossAndRecording,
    TestResult,
    Weight,
    Spike,
)


def max_over_time(output: LIFState) -> jax.Array:
    return jnp.max(output.V, axis=0)


def nll_loss(output: jax.Array, targets: jax.Array) -> float:
    n_classes = targets.shape[0]
    idx = jnp.argmin(targets)
    targets = jnp.array(idx == jnp.arange(n_classes))
    output = jnp.maximum(output, 0)
    preds = jax.nn.log_softmax(output)
    loss = -jnp.sum(targets * preds)
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


def loss_wrapper(  # pylint: disable=too-many-arguments,too-many-locals
    apply_fn: Apply,
    loss_fn: Callable[[jax.Array, jax.Array, float], float],
    tau_mem: float,
    n_neurons: int,
    n_outputs: int,
    weights: List[Weight],
    batch: Tuple[EventPropSpike, jax.Array],
    vmap: bool = True,
    external: Optional[List[Spike]] = None,
    carry: Optional[Any] = None,
) -> LossAndRecording:
    input_spikes, target = batch

    if vmap:
        # Check if run with known spikes
        if external is None:
            in_axes = (None, 0, None, None)
        else:
            in_axes = (None, 0, 0, None)
        # Create batched functions
        apply_fn = jax.vmap(
            apply_fn, in_axes=in_axes
        )

        first_spike_function = jax.vmap(
            first_spike, in_axes=(0, None, None)
        )

        loss_function = jax.vmap(
            loss_fn, in_axes=(0, 0, None)
        )
    else:
        first_spike_function = first_spike
        loss_function = loss_fn

    _, _, output, recording = apply_fn(
        weights,
        input_spikes,
        external,
        carry
    )

    t_first_spike = first_spike_function(output, n_neurons, n_outputs)

    loss_value = loss_function(t_first_spike, target, tau_mem)

    if vmap:
        loss_value = jnp.mean(loss_value)

    return loss_value, (t_first_spike, recording)


def target_time_loss(
    first_spikes: jax.Array, target: jax.Array, tau_mem: float
) -> float:
    loss_value = -jnp.sum(
        jnp.log(1 + jnp.exp(-jnp.abs(first_spikes - target) / tau_mem))
    )
    return loss_value


def ttfs_loss(
    first_spikes: jax.Array, target: jax.Array, tau_mem: float
) -> float:
    idx = jnp.argmin(target)
    first_spikes = jnp.minimum(jnp.abs(first_spikes), 2 * tau_mem)
    return -jnp.log(
        jnp.sum(jnp.exp((first_spikes[idx] - first_spikes) / tau_mem))
    )


def mse_loss(
    first_spikes: jax.Array, target: jax.Array, tau_mem: float
) -> float:
    return jnp.sum(
        jnp.square((jnp.minimum(first_spikes, 2 * tau_mem) - target) / tau_mem)
    )


def first_spike(
    spikes: EventPropSpike,
    size: int,
    n_outputs: int
) -> jax.Array:
    return jnp.array(
        [
            jnp.min(jnp.where(spikes.idx == idx, spikes.time, jnp.inf))
            for idx in range(size)
        ][size - n_outputs:]
    )


def loss_and_acc(
    loss_fn: Callable,
    weights: List[Weight],
    dataset: Tuple[EventPropSpike, jax.Array],
) -> TestResult:
    batched_loss = jax.vmap(loss_fn, in_axes=(None, 0))
    loss, (t_first_spike, recording) = batched_loss(weights, dataset)
    accuracy = jnp.argmin(dataset[1], axis=-1) == jnp.argmin(
        t_first_spike, axis=-1
    )
    return TestResult(
        jnp.mean(loss),
        jnp.mean(accuracy),
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
    accuracy = jnp.argmin(dataset[1], axis=-1) == jnp.argmin(
        t_first_spike, axis=-1
    )
    return TestResult(
        jnp.mean(loss),
        jnp.mean(accuracy),
        t_first_spike,
        recording,
    )
