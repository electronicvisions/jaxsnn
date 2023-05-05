import jax
import jax.numpy as np
from jax import jit


@jit
def decode(x):
    x = np.max(x, 0)
    log_p_y = jax.nn.log_softmax(x, axis=1)
    return log_p_y


def MaxOverTimeDecode():
    def init_fn(rng, input_shape):
        return (input_shape, None, rng)

    def apply_fn(params, inputs, **kwargs):
        return decode(inputs), None

    return init_fn, apply_fn
