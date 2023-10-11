import jax
import jax.numpy as np


@jax.jit
def decode(inputs):
    inputs = np.max(inputs, 0)
    log_p_y = jax.nn.log_softmax(inputs, axis=1)
    return log_p_y


def max_over_time_decode():
    def init_fn(rng, input_shape):
        return (input_shape, None, rng)

    def apply_fn(weights, inputs, **kwargs):  # pylint: disable=unused-argument
        return decode(inputs), None

    return init_fn, apply_fn
