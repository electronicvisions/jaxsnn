# pylint: disable=logging-fstring-interpolation
from typing import Any, Callable, Tuple, List

import jax
import jax.numpy as jnp
import optax
import jaxsnn
from jaxsnn.base.params import LIFParameters
from jaxsnn.base.dataset import data_loader
from jaxsnn.event.types import OptState, Spike
from jaxsnn.event.utils import time_it


log = jaxsnn.get_logger("jaxsnn.event.training")


def update(
    optimizer,
    loss_fn: Callable,
    params: LIFParameters,
    state: OptState,
    batch: Tuple[Spike, jax.Array],
) -> Tuple[OptState, Tuple[jax.Array, jax.Array]]:
    value, grad = jax.value_and_grad(loss_fn, has_aux=True)(
        state.weights, batch
    )
    grad = jax.tree_util.tree_map(lambda g: g / params.tau_syn, grad)

    updates, opt_state = optimizer.update(grad, state.opt_state)
    weights = optax.apply_updates(state.weights, updates)
    return OptState(opt_state, weights, state.rng), (value, grad)


def epoch(
    update_fn: Callable,
    test_fn: Callable[
        [List[jax.Array], Tuple[jax.Array, jax.Array]],
        Tuple[Any, str]
    ],
    trainset,
    testset,
    opt_state: OptState,
    i: int,
):  # pylint: disable=too-many-arguments, too-many-locals
    rng, train_rng, test_rng = jax.random.split(opt_state.rng, 3)
    trainset_batched = data_loader(trainset, 64, rng=train_rng)
    res, duration = time_it(
        jax.lax.scan,
        update_fn,
        opt_state,
        trainset_batched
    )
    opt_state, (recording, grad) = res
    opt_state = OptState(opt_state.opt_state, opt_state.weights, rng)

    testset_batched = data_loader(testset, 64, rng=test_rng)
    test_result, test_str = test_fn(opt_state.weights, testset_batched)
    mean_spikes = jnp.sum(recording[1][1][0].idx >= 0, axis=-1).mean()
    mean_grad = grad[0].input.mean()
    log.info(
        f"Epoch {i}, test: {test_str}, spikes: {mean_spikes:.1f}, "
        f"grad: {mean_grad:.5f}, time: {duration:.2f}s"
    )
    return opt_state, (test_result, opt_state.weights, duration)
