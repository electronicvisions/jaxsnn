from typing import Tuple

import jax.numpy as np

from jaxsnn.functional.encode import one_hot
from jaxsnn.types import Array


def nll_loss(
    snn_apply, params, batch, expected_spikes=0.5, rho=1e-4
) -> Tuple[float, Array]:
    inputs, targets = batch
    preds, recording = snn_apply(params, inputs)
    targets = one_hot(targets, preds.shape[1])
    loss = -np.mean(np.sum(targets * preds, axis=1))
    regularization = rho * np.sum(
        np.square(np.sum(recording[1].z, axis=0) - expected_spikes)
    )
    return loss + regularization, recording


def acc_and_loss(snn_apply, params, batch):
    inputs, targets = batch
    preds, _ = snn_apply(params, inputs)
    correct = (np.argmax(preds, axis=1) == targets).sum()
    accuracy = correct / len(targets)

    targets = one_hot(targets, preds.shape[1])
    loss = -np.mean(np.sum(targets * preds, axis=1))
    return np.mean(accuracy), np.mean(loss)
