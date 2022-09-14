from typing import Tuple

import jax.numpy as jnp
from jaxsnn.functional.encode import one_hot


def nll_loss(snn_apply, params, batch) -> Tuple[float, jnp.DeviceArray]:
    inputs, targets = batch
    preds, recording = snn_apply(params, inputs), jnp.empty((0))
    targets = one_hot(targets, preds.shape[1])
    return -jnp.mean(jnp.sum(targets * preds, axis=1)), recording


def acc_and_loss(snn_apply, params, batch):
    inputs, targets = batch
    preds, _ = snn_apply(params, inputs), jnp.empty((0))
    correct = (jnp.argmax(preds, axis=1) == targets).sum()
    accuracy = correct / len(targets)

    # also calculate loss
    targets = one_hot(targets, preds.shape[1])
    loss = -jnp.mean(jnp.sum(targets * preds, axis=1))
    return jnp.mean(accuracy), jnp.mean(loss)
