# Copyright (c) 2022 Heidelberg University. All rights reserved.
#
# Released under Apache 2.0 license as described in the file LICENSE.
# Authors: Christian Pehle

import tree_math
from jaxsnn.base.types import ArrayLike
from typing import TypeVar, Generic, Tuple
import jax.numpy as jnp

Output = TypeVar("Output")


@tree_math.struct
class State:
    queue: ArrayLike
    head: int
    used: int


class ArrayQueue:
    def enqueue(state: State, data: ArrayLike) -> State:
        queue = state.queue.at[(state.head + state.used) % state.queue.shape[0]].set(
            data
        )
        return State(queue=queue, head=state.head, used=state.used + 1)

    def dequeue(state: State) -> Tuple[Output, State]:
        x = state.queue[state.head]
        return (
            x,
            State(
                queue=state.queue.at[state.head].set(jnp.zeros_like(x)),
                head=(state.head + 1) % state.queue.shape[0],
                used=state.used - 1,
            ),
        )
