from typing import Tuple

import jax
import jax.numpy as jnp
from jaxsnn.discrete.encode import one_hot


def nll_loss(  # pylint: disable=too-many-arguments
    snn_apply, weights, batch, decoder, expected_spikes=0.5, rho=1e-4,
) -> Tuple[float, jax.Array]:
    inputs, targets = batch
    _, _, preds, recording = snn_apply(weights, inputs, None, None)
    preds_decoded = decoder(preds)
    targets = one_hot(targets, preds_decoded.shape[1])
    loss = -jnp.mean(jnp.sum(targets * preds_decoded, axis=1))
    regularization = rho * jnp.sum(
        jnp.square(jnp.sum(recording[0].z, axis=0) - expected_spikes)
    )
    return loss + regularization, recording


def acc_and_loss(snn_apply, weights, batch, decoder):
    inputs, targets = batch
    _, _, preds, _ = snn_apply(weights, inputs, None, None)
    preds_decoded = decoder(preds)
    correct = (jnp.argmax(preds_decoded, axis=1) == targets).sum()
    accuracy = correct / len(targets)

    targets = one_hot(targets, preds_decoded.shape[1])
    loss = -jnp.mean(jnp.sum(targets * preds_decoded, axis=1))
    return jnp.mean(accuracy), jnp.mean(loss)
