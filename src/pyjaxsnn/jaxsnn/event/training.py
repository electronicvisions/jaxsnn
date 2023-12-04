# pylint: disable=logging-fstring-interpolation
import logging
from typing import Callable, Tuple

import jax
import jax.numpy as np
import optax
from jaxsnn.base.params import LIFParameters
from jaxsnn.event.dataset import Dataset
from jaxsnn.event.loss import loss_and_acc
from jaxsnn.event.types import LossFn, OptState, Spike
from jaxsnn.event.utils import time_it

log = logging.getLogger("root")


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
    trainset: Dataset,
    testset: Dataset,
    opt_state: OptState,
    i: int,
):  # pylint: disable=too-many-arguments
    res, duration = time_it(jax.lax.scan, update_fn, opt_state, trainset[:2])
    opt_state, (recording, grad) = res

    test_result = loss_and_acc(loss_fn, opt_state.weights, testset[:2])
    log.info(
        f"Epoch {i}, "
        f"loss: {test_result[0]:.4f}, "
        f"acc: {test_result[1]:.3f}, "
        f"spikes: {np.sum(recording[1][1][0].idx >= 0, axis=-1).mean():.1f}, "
        f"grad: {grad[0].input.mean():.5f}, "
        f"in {duration:.2f} s"
    )
    return opt_state, (test_result, opt_state.weights, duration)
