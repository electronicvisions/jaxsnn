import jax
import jax.numpy as np


@jax.jit
def max_over_time_decode(inputs):
    inputs = np.max(inputs, 0)
    log_p_y = jax.nn.log_softmax(inputs, axis=1)
    return log_p_y
