from typing import List

import jax
import jax.numpy as jnp
from jax import random
from jaxsnn.dataset.yingyang import YinYangDataset
from jaxsnn.functional.encode import spatio_temporal_encode
from jaxsnn.functional.leaky_integrator import li_feed_forward_step, li_init_state, li_init_weights
from jaxsnn.functional.lif import lif_init_state, lif_init_weights, lif_step


def decode(x):
    x = jnp.maximum(x, 0)
    log_p_y = jax.nn.softmax(x, axis=0)  # change to axis = 1 if batch?
    return log_p_y


# do multiple time steps, later convert to one time step + scan function
def forward(states: List, weights: List, input_values: jnp.DeviceArray):
    voltages = []
    lif_state, li_state = states
    for ts in range(input_values.shape[0]):
        lif_state, z = lif_step(lif_state, *weights[0], input_values[ts])
        li_state, voltage = li_feed_forward_step(li_state, *weights[1], z)
        log_p_y = decode(voltage)
        voltages.append(log_p_y)
    return voltages


if __name__ == "__main__":
    key = random.PRNGKey(42)
    dataset = YinYangDataset(key, 1000)
    states = [lif_init_state(50), li_init_state(3)]
    weights = [lif_init_weights(key, 4, 50), li_init_weights(key, 50, 3)]

    for input, output in dataset:
        input = spatio_temporal_encode(input, 10)
        forward(states, weights, input)

# TODO: speed up dataset, clean code, add tests for encode, update encode
