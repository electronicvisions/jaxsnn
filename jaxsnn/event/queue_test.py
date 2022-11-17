# Copyright (c) 2022 Heidelberg University. All rights reserved.
#
# Released under Apache 2.0 license as described in the file LICENSE.
# Authors: Christian Pehle

from jaxsnn.event.queue import ArrayQueue, State
import jax.numpy as jnp
import numpy as np


def test_enqueue_dequeue():
    state = State(queue=jnp.zeros((100, 2)), head=0, used=0)

    for i in range(5):
        state = ArrayQueue.enqueue(state, jnp.full(2, i))

    for i in range(5):
        out, state = ArrayQueue.dequeue(state)
        np.testing.assert_array_equal(out, jnp.full(2, i))
