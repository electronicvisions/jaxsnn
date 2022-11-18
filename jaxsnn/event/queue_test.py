# Copyright (c) 2022 Heidelberg University. All rights reserved.
#
# Released under Apache 2.0 license as described in the file LICENSE.
# Authors: Christian Pehle

from jaxsnn.event.queue import ArrayQueue, State
import jax
import jax.numpy as jnp
import numpy as np


def test_enqueue_dequeue():
    state = State(queue=jnp.zeros((100, 2)), head=0, used=0)

    for i in range(5):
        state = ArrayQueue.enqueue(state, jnp.full(2, i))

    for i in range(5):
        out, state = ArrayQueue.dequeue(state)
        np.testing.assert_array_equal(out, jnp.full(2, i))


def test_enqueue_has_grad():
    state = State(queue=jnp.zeros((100, 2)), head=0, used=0)

    def f(state, x):
        state = ArrayQueue.enqueue(state, x)
        return state.queue[0][0]

    grad = jax.grad(f, allow_int=True)(state, np.random.randn(2))
    print(grad)


def test_dequeue_has_grad():
    state = State(queue=jnp.zeros((100, 2)), head=0, used=0)
    state = ArrayQueue.enqueue(state, jnp.full(2, 1.0))

    def f(state):
        output, state = ArrayQueue.dequeue(state)
        return output[0]

    grad = jax.grad(f, allow_int=True)(state)
    print(grad)
