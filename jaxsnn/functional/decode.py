import jax
import jax.numpy as np
from jax import jit


@jit
def decode(x):
    x = np.max(x, 0)
    log_p_y = jax.nn.log_softmax(x, axis=1)
    return log_p_y
