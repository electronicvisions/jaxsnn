from typing import List, Tuple

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
from jax.lax import scan
from jax.tree_util import tree_map
from jaxsnn.functional.encode import one_hot
from jaxsnn.functional.leaky_integrator import (
    li_feed_forward_step,
    li_init_state,
    li_integrate,
)
from jaxsnn.functional.lif import lif_init_state, lif_integrate, lif_step


def decode(x):
    x = jnp.max(x, 0)
    log_p_y = jax.nn.log_softmax(x, axis=1)
    return log_p_y


@jit
def update(
    params,
    last_grads,
    state,
    x,
    y,
    momentum: float,
    step_size: float,
):
    # at the moment we do not take grad of state or grad of topology
    (loss_value, recording), grads = value_and_grad(loss, has_aux=True)(
        params, state, (x, y)
    )
    params = tree_map(
        lambda p, g, lg: p - step_size * (g + momentum * lg),
        params,
        grads,
        last_grads,
    )
    return params, grads, loss_value, recording


def loss(params, state, batch) -> Tuple[float, jnp.DeviceArray]:
    inputs, targets = batch
    preds, recording = forward(params, state, inputs)
    return -jnp.mean(jnp.sum(targets * preds, axis=1)), recording


def accuracy_and_loss(params, state, batch):
    inputs, targets = batch
    preds, _ = forward(params, state, inputs)
    correct = (jnp.argmax(preds, axis=1) == jnp.argmax(targets, axis=1)).sum()
    accuracy = correct / len(targets)
    loss = -jnp.mean(jnp.sum(targets * preds, axis=1))
    return accuracy, loss


def forward(params, states, input_values) -> Tuple:
    _, spikes = lif_integrate((states[0], *params[0]), input_values)
    _, voltages = li_integrate((states[1], *params[1]), spikes)
    log_p_y = decode(voltages)
    return log_p_y, (voltages, spikes)
