from typing import Tuple

import jax
import jax.numpy as np
from jaxsnn.discrete.encode import one_hot


def nll_loss(
    snn_apply, weights, batch, expected_spikes=0.5, rho=1e-4
) -> Tuple[float, jax.Array]:
    inputs, targets = batch
    preds, recording = snn_apply(weights, inputs)
    targets = one_hot(targets, preds.shape[1])
    loss = -np.mean(np.sum(targets * preds, axis=1))
    regularization = rho * np.sum(
        np.square(np.sum(recording[1].z, axis=0) - expected_spikes)
    )
    return loss + regularization, recording


def acc_and_loss(snn_apply, weights, batch):
    inputs, targets = batch
    preds, _ = snn_apply(weights, inputs)
    correct = (np.argmax(preds, axis=1) == targets).sum()
    accuracy = correct / len(targets)

    targets = one_hot(targets, preds.shape[1])
    loss = -np.mean(np.sum(targets * preds, axis=1))
    return np.mean(accuracy), np.mean(loss)
