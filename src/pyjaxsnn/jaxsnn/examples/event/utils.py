from typing import Any, Callable, List, Tuple, Optional

import jax
import jax.numpy as jnp
from jaxsnn.event.loss import first_spike
from jaxsnn.event.types import (
    Apply,
    EventPropSpike,
    Weight,
    Spike,
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


def loss_wrapper(
    apply_fn: Apply,
    loss_fn: Callable[[jax.Array, jax.Array, float], float],
    tau_mem: float,
    n_neurons: int,
    n_outputs: int,
    weights: List[Weight],
    batch: Tuple[EventPropSpike, jax.Array],
    external: Optional[List[Spike]] = None,
    carry: Optional[Any] = None,
) -> Tuple[jax.Array, Tuple[jax.Array, Any]]:
    input_spikes, target = batch

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

    _, _, output, recording = apply_fn(
        weights,
        input_spikes,
        external,
        carry
    )

    t_first_spike = first_spike_function(output, n_neurons, n_outputs)

    loss_value = jnp.mean(loss_function(t_first_spike, target, tau_mem))

    return loss_value, (t_first_spike, recording)
