from typing import Tuple

import jax.numpy as jnp
from jaxsnn.functional.encode import one_hot


def nll_loss(
    snn_apply, params, batch, expected_spikes=0.5, rho=1e-4
) -> Tuple[float, jnp.DeviceArray]:
    inputs, targets = batch
    preds, recording = snn_apply(params, inputs)
    targets = one_hot(targets, preds.shape[1])
    loss = -jnp.mean(jnp.sum(targets * preds, axis=1))
    regularization = rho * jnp.sum(
        jnp.square(jnp.sum(recording[1].z, axis=0) - expected_spikes)
    )
    return loss + regularization, recording


def acc_and_loss(snn_apply, params, batch):
    inputs, targets = batch
    preds, _ = snn_apply(params, inputs)
    correct = (jnp.argmax(preds, axis=1) == targets).sum()
    accuracy = correct / len(targets)

    targets = one_hot(targets, preds.shape[1])
    loss = -jnp.mean(jnp.sum(targets * preds, axis=1))
    return jnp.mean(accuracy), jnp.mean(loss)
