"""Implement functionality of lax for easier debugging"""
import logging
from typing import Callable, Optional

import jax
from jax.tree_util import tree_flatten, tree_map, tree_unflatten

log = logging.getLogger("root")


def scan(
    inner_fn: Callable,
    init,
    inputs,
    length: Optional[int] = None,
    reverse: bool = False,
):
    xs_flat, xs_tree = tree_flatten(inputs)
    carry = init
    outputs = []
    length = len(xs_flat[0])
    for i in range(length):
        if reverse:
            i = length - i - 1
        xs_slice = [x[i] for x in xs_flat]
        carry, output = inner_fn(carry, tree_unflatten(xs_tree, xs_slice))
        outputs.append(output)

    def stack(*args):
        return jax.numpy.stack(args)

    stacked_y = tree_map(  # pylint: disable=no-value-for-parameter
        stack, *outputs
    )
    return carry, stacked_y


def cond(pred: bool, true_fun: Callable, false_fun: Callable, *operands):
    """Call both function to evaluate compiled behaviour of `jax.lax.scan`"""
    true_result = true_fun(*operands)
    false_result = false_fun(*operands)
    return jax.tree_map(
        lambda t_item, f_item: jax.lax.select(pred, t_item, f_item),
        true_result,
        false_result,
    )
