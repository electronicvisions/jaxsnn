from typing import Any, Callable, List, Tuple, Optional

import jax
import jax.numpy as jnp
from jaxsnn.base.types import Parameters
from jaxsnn.event.loss import first_spike, rate_loss
from jaxsnn.event.types import (
    ModelApplyFn,
    Observables,
    Spike,
    IOData,
)


def test_step(
    loss_fn: Callable,
    weights: List[jax.Array],
    dataset: Tuple[jax.Array, jax.Array],
) -> Tuple[Tuple[jax.Array, jax.Array, Any], str]:

    batched_loss_fn = jax.vmap(loss_fn, in_axes=(None, 0))
    loss, (t_first_spike, recording) = batched_loss_fn(weights, dataset)
    accuracy = jnp.argmin(dataset[1], axis=-1) == jnp.argmin(
        t_first_spike, axis=-1
    )

    loss = jnp.mean(loss)
    accuracy = jnp.mean(accuracy)
    test_str = f"loss: {loss:.4f}, acc: {accuracy:.3f}"

    return (loss, accuracy, t_first_spike, recording), test_str


def test_step2(
    loss_fn: Callable,
    weights: List[jax.Array],
    dataset: Tuple[jax.Array, jax.Array],
) -> Tuple[Tuple[jax.Array, jax.Array, Any], str]:

    batched_loss_fn = jax.vmap(loss_fn, in_axes=(None, 0))
    loss, (n_spike_count, recording) = batched_loss_fn(weights, dataset)
    accuracy = dataset[1] == jnp.argmax(n_spike_count, axis=-1)

    loss = jnp.mean(loss)
    accuracy = jnp.mean(accuracy)
    test_str = f"loss: {loss:.4f}, acc: {accuracy:.3f},"

    return (loss, accuracy, n_spike_count, recording), test_str


def loss_wrapper(
    apply_fn: ModelApplyFn,
    loss_fn: Callable[[jax.Array, jax.Array, float], float],
    tau_mem: float,
    output_node: str,
    n_outputs: int,
    parameters: Parameters,
    batch: Tuple[IOData, jax.Array],
    carry: Optional[Any] = None,
) -> Tuple[jax.Array, Tuple[jax.Array, Observables]]:
    input_spikes, target = batch

    first_spike_function = jax.vmap(
        first_spike, in_axes=(0, None)
    )

    loss_function = jax.vmap(
        loss_fn, in_axes=(0, 0, None)
    )

    observables = apply_fn(
        input_spikes,
        parameters,
    )

    output_spikes = observables[output_node].spike
    t_first_spike = first_spike_function(output_spikes, n_outputs)

    loss_value = jnp.mean(loss_function(t_first_spike, target, tau_mem))

    return loss_value, (t_first_spike, observables)


def loss_wrapper2(
    apply_fn: ModelApplyFn,
    loss_fn: Callable[[jax.Array, jax.Array, float], float],
    tau_mem: float,
    output_node: str,
    n_outputs: int,
    parameters: Parameters,
    batch: Tuple[IOData, jax.Array],
    carry: Optional[Any] = None,
) -> Tuple[jax.Array, Tuple[jax.Array, IOData]]:
    input_spikes, target = batch
    target_one_hot = jax.nn.one_hot(target, n_outputs)
    summed_spike_times_function = jax.vmap(
        summed_spike_times, in_axes=(0, None)
    )
    loss_function = jax.vmap(
        loss_fn, in_axes=(0, 0, None)
    )

    observables = apply_fn(
        input_spikes,
        parameters,
    )

    output_spikes = observables[output_node].spike
    output_rate = summed_spike_times_function(output_spikes, n_outputs)
    loss_values = loss_function(output_rate, target_one_hot, tau_mem)
    loss_value = jnp.mean(loss_values)
    return loss_value, (output_rate, observables)
