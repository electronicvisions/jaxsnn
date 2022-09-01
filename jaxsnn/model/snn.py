from typing import List

import jax
import jax.numpy as jnp
from jax import value_and_grad
from jax.tree_util import tree_map
from jaxsnn.functional.leaky_integrator import (li_feed_forward_step,
                                                li_init_state)
from jaxsnn.functional.lif import lif_init_state, lif_step


def decode(x):
    x = jnp.max(x, 0)
    log_p_y = jax.nn.softmax(x, axis=1)
    return log_p_y


def update(
    params,
    x: jnp.DeviceArray,
    y: jnp.DeviceArray,
    # last_grads,
    # momentum_mass: float,
    step_size: float,
):
    loss_value, grads = value_and_grad(loss)(params, (x, y))
    a = 4
    return tree_map(
        lambda p, g: p - step_size * g,  # (g + momentum_mass * lg),
        params,
        grads,
        # last_grads,
    ), loss_value


def loss(params: List, batch):
    inputs, targets = batch
    preds = forward(params, inputs)
    return -jnp.mean(jnp.sum(targets * preds, axis=1))


# do multiple time steps, later convert to one time step + scan function
def forward(params: List, input_values: jnp.DeviceArray):
    batch_size = input_values.shape[1]
    states = [lif_init_state((batch_size, 50)), li_init_state((batch_size, 3))]
    voltages = []
    lif_state, li_state = states
    for ts in range(input_values.shape[0]):
        lif_state, z = lif_step(lif_state, *params[0], input_values[ts])
        if jnp.count_nonzero(z) > 0:
            print(f"Spike count: {jnp.count_nonzero(z)}")
        li_state, voltage = li_feed_forward_step(li_state, *params[1], z)
        voltages.append(voltage)
    log_p_y = decode(jnp.array(voltages))
    return log_p_y
