"""Implement functionality of lax for easier debugging"""
import logging
from typing import Callable, Optional

import jax
from jax.tree_util import tree_flatten, tree_map, tree_unflatten

log = logging.getLogger("root")


def scan(
    f: Callable,
    init,
    xs,
    length: Optional[int] = None,
    reverse: bool = False,
    print_bool: bool = False,
):
    xs_flat, xs_tree = tree_flatten(xs)
    carry = init
    ys = []
    length = len(xs_flat[0])
    for i in range(length):
        if print_bool:
            log.info(f"Sample {i}")
        if reverse:
            i = length - i - 1
        xs_slice = [x[i] for x in xs_flat]
        carry, y = f(carry, tree_unflatten(xs_tree, xs_slice))
        ys.append(y)

    def stack(*ys):
        return jax.numpy.stack(ys)

    stacked_y = tree_map(stack, *ys)
    return carry, stacked_y


def cond(pred: bool, true_fun: Callable, false_fun: Callable, *operands):
    """Call both function to evaluate compiled behaviour of `jax.lax.scan`"""
    true_result = true_fun(*operands)
    false_result = false_fun(*operands)
    return jax.tree_map(
        lambda t, f: jax.lax.select(pred, t, f), true_result, false_result
    )
