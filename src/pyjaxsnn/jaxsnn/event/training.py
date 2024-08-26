# pylint: disable=logging-fstring-interpolation
from typing import Callable, Tuple

import jax
import jax.numpy as np
import optax
import jaxsnn
from jaxsnn.base.params import LIFParameters
from jaxsnn.event.dataset import data_loader
from jaxsnn.event.loss import loss_and_acc
from jaxsnn.event.types import LossFn, OptState, Spike
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
    return OptState(opt_state, weights), (value, grad)


def epoch(
    update_fn: Callable,
    loss_fn: LossFn,
    trainset,
    testset,
    opt_state: OptState,
    i: int,
):  # pylint: disable=too-many-arguments
    rng = jax.random.PRNGKey(i)
    trainset_batched = data_loader(trainset, 64, rng)
    res, duration = time_it(
        jax.lax.scan,
        update_fn,
        opt_state,
        trainset_batched
    )
    opt_state, (recording, grad) = res

    testset_batched = data_loader(testset, 64)
    test_result = loss_and_acc(loss_fn, opt_state.weights, testset_batched)
    log.info(
        f"Epoch {i}, "
        f"loss: {test_result[0]:.4f}, "
        f"acc: {test_result[1]:.3f}, "
        f"spikes: {np.sum(recording[1][1][0].idx >= 0, axis=-1).mean():.1f}, "
        f"grad: {grad[0].input.mean():.5f}, "
        f"in {duration:.2f} s"
    )
    return opt_state, (test_result, opt_state.weights, duration)
