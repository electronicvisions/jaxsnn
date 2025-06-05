import jax
import jax.numpy as jnp

from jaxsnn.discrete.types import DenseData


@jax.jit
def max_over_time_decode(inputs: DenseData) -> jnp.ndarray:
    """
    Decode the output of a jaxsnn model by taking the maximum over time and
    applying a log softmax.

    :params inputs: DenseData of shape (n_batch_size, n_time_steps, n_neurons)
        representing the output of a jaxsnn model.
    :return: jnp.ndarray of shape (n_batch_size, n_neurons) representing the
        decoded output.
    """
    inputs = jnp.max(inputs, axis=0)
    log_p_y = jax.nn.log_softmax(inputs, axis=0)
    return log_p_y
