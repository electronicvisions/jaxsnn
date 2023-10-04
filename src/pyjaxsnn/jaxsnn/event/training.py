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
    p: LIFParameters,
    state: OptState,
    batch: Tuple[Spike, jax.Array],
) -> Tuple[OptState, Tuple[jax.Array, jax.Array]]:
    value, grad = jax.value_and_grad(loss_fn, has_aux=True)(state.params, batch)
    grad = jax.tree_util.tree_map(lambda g: g / p.tau_syn, grad)

    updates, opt_state = optimizer.update(grad, state.opt_state)
    params = optax.apply_updates(state.params, updates)
    return OptState(opt_state, params), (value, grad)


def epoch(
    update_fn: Callable,
    loss_fn: LossFn,
    trainset: Dataset,
    testset: Dataset,
    opt_state: OptState,
    i: int,
):
    res, duration = time_it(jax.lax.scan, update_fn, opt_state, trainset[:2])
    opt_state, (recording, grad) = res

    test_result = loss_and_acc(loss_fn, opt_state.params, testset[:2])
    log.info(
        f"Epoch {i}, "
        f"loss: {test_result[0]:.4f}, "
        f"acc: {test_result[1]:.3f}, "
        f"spikes: {np.sum(recording[1][1][0].idx >= 0, axis=-1).mean():.1f}, "
        f"grad: {grad[0].input.mean():.5f}, "
        f"in {duration:.2f} s"
    )
    return opt_state, (test_result, opt_state.params, duration)
