import jax
import jax.numpy as jnp


@jax.jit
def max_over_time_decode(inputs):
    inputs = jnp.max(inputs, 0)
    log_p_y = jax.nn.log_softmax(inputs, axis=1)
    return log_p_y
