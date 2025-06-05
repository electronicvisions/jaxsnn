# pylint: disable=logging-fstring-interpolation
from typing import Any, Callable, Tuple, List

import jax
import jax.numpy as jnp
import optax
import jaxsnn
from jaxsnn.base.dataset import data_loader
from jaxsnn.event.types import OptState, IOData, Parameters
from jaxsnn.event.utils import time_it


log = jaxsnn.get_logger("jaxsnn.event.training")


def update(
    optimizer,
    loss_fn: Callable,
    state: OptState,
    batch: Tuple[IOData, jax.Array],
) -> Tuple[OptState, Tuple[jax.Array, Parameters]]:
    value, grad = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params,
        batch,
    )
    updates, opt_state = optimizer.update(grad, state.opt_state)
    params = optax.apply_updates(state.params, updates)
    return OptState(opt_state, params, state.rng), (value, grad)


def epoch(
    update_fn: Callable,
    test_fn: Callable[
        [List[jax.Array], Tuple[jax.Array, jax.Array]],
        Tuple[Any, str]
    ],
    trainset,
    testset,
    batch_size_train: int,
    batch_size_test: int,
    state: OptState,
    i: int,
):  # pylint: disable=too-many-arguments, too-many-locals
    rng, train_rng, test_rng = jax.random.split(state.rng, 3)
    trainset_batched = data_loader(trainset, batch_size_train, rng=train_rng)
    res, duration = time_it(
        jax.lax.scan, update_fn, state, trainset_batched
    )
    state, (recording, grad) = res
    state = OptState(state.opt_state, state.params, rng)

    testset_batched = data_loader(testset, batch_size_test, rng=test_rng)
    test_result, test_str = test_fn(state.params, testset_batched)

    spikes_info = ""
    for node in recording[1][1]:
        spikes = recording[1][1][node]
        spikes = spikes.where(spikes.internal) if spikes is not None else None
        spikes_mean = (
            f"{jnp.sum(spikes.idx >= 0, axis=-1).mean():.4f}"
            if spikes is not None
            else None
        )
        spikes_info += f"\t\t{node}: {spikes_mean}\n"

    grads_info = ""
    for node in recording[1][1]:
        grad_node = grad[node]
        grad_mean = f"{grad_node.mean():.8f}" \
            if grad_node is not None else None
        grads_info += f"\t\t{node}: {grad_mean}\n"

    log.info(
        f"Epoch {i}:\n"
        f"\ttest result:\n\t\t{test_str},\n"
        f"\tspikes:\n{spikes_info}"
        f"\tgrad:\n{grads_info}"
        f"\tin {duration:.2f}s"
    )
    return (
        OptState(state.opt_state, state.params, rng),
        (test_result, state.params, duration))
