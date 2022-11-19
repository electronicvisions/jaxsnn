# Copyright (c) 2022 Heidelberg University. All rights reserved.
#
# Released under Apache 2.0 license as described in the file LICENSE.
# Authors: Christian Pehle

from jaxsnn.base.types import ArrayLike
import tree_math
import jax.numpy as jnp
import jax
import dataclasses


@dataclasses.dataclass
@tree_math.struct
class State:
    a: ArrayLike
    b: ArrayLike
    fa: ArrayLike
    fb: ArrayLike


def illinois_method(f, a, b, eps):
    fa = f(a)
    fb = f(b)
    assert jnp.abs(fa) <= jnp.abs(fb)
    init = State(a=a, b=b, fa=fa, fb=fb)

    def cond(state: State):
        return jnp.abs(state.b - state.a) > eps

    def body_fun(state: State):
        a = state.a
        b = state.b
        fa = state.fa
        fb = state.fb
        c = a - (fa * (b - a)) / (fb - fa)
        fc = f(c)
        b, fb = jax.lax.cond(fa * fc <= 0, lambda: (a, fa), lambda: (b, 0.5 * fb))
        a = c
        fa = fc
        return State(a=a, b=b, fa=fa, fb=fb)

    return jax.lax.while_loop(cond, body_fun, init)
