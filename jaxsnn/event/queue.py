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


@jax.custom_vjp
def enqueue(state: State, data: Any) -> State:
    queue = state.queue.at[(state.head + state.used) % state.queue.shape[0]].set(data)
    return State(queue=queue, head=state.head, used=state.used + 1)


@jax.custom_vjp
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


def enqueue_fwd(state: State, data: Any):
    return enqueue(state, data), None


def enqueue_bwd(_, state: State):
    grad, state = dequeue(state)
    return state, grad


def dequeue_fwd(state: State):
    return dequeue(state), None


def dequeue_bwd(_, grad):
    doutput, state = grad
    state = enqueue(state, doutput)
    return (state,)


enqueue.defvjp(enqueue_fwd, enqueue_bwd)
dequeue.defvjp(dequeue_fwd, dequeue_bwd)
