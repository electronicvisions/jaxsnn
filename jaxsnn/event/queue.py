# Copyright (c) 2022 Heidelberg University. All rights reserved.
#
# Released under Apache 2.0 license as described in the file LICENSE.
# Authors: Christian Pehle

import tree_math
from typing import Tuple, Any
import jax.numpy as jnp
import jax

import dataclasses


@dataclasses.dataclass
@tree_math.struct
class State:
    queue: Any  # TODO: Should be something more specific
    head: int
    used: int


class ArrayQueue:
    @jax.custom_vjp
    @staticmethod
    def enqueue(state: State, data: Any) -> State:
        queue = state.queue.at[(state.head + state.used) % state.queue.shape[0]].set(
            data
        )
        return State(queue=queue, head=state.head, used=state.used + 1)

    @jax.custom_vjp
    @staticmethod
    def dequeue(state: State) -> Tuple[Any, State]:
        x = state.queue[state.head]
        return (
            x,
            State(
                queue=state.queue.at[state.head].set(jnp.zeros_like(x)),
                head=(state.head + 1) % state.queue.shape[0],
                used=state.used - 1,
            ),
        )

    @staticmethod
    def enqueue_fwd(state: State, data: Any):
        return ArrayQueue.enqueue(state, data), None

    def enqueue_bwd(_, state: State):
        grad, state = ArrayQueue.dequeue(state)
        return state, grad

    @staticmethod
    def dequeue_fwd(state: State):
        return ArrayQueue.dequeue(state), None

    def dequeue_bwd(_, grad):
        doutput, state = grad
        state = ArrayQueue.enqueue(state, doutput)
        return (state,)


ArrayQueue.enqueue.defvjp(ArrayQueue.enqueue_fwd, ArrayQueue.enqueue_bwd)
ArrayQueue.dequeue.defvjp(ArrayQueue.dequeue_fwd, ArrayQueue.dequeue_bwd)
