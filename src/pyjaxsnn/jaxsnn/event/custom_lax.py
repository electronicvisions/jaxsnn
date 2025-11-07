"""Implement functionality of lax for easier debugging"""
from typing import Callable, Sequence, Optional

import jax
import jax.numpy as jnp

from jaxsnn.event.utils import is_branch_free

try:
    from jax.tree import flatten as tree_flatten
    from jax.tree import unflatten as tree_unflatten
    from jax.tree import map as tree_map
except ImportError:
    # for compatibility with jax@:0.4.25
    from jax.tree_util import tree_flatten
    from jax.tree_util import tree_unflatten
    from jax.tree_util import tree_map


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


def cond(
    pred: bool,
    true_fun: Callable,
    false_fun: Callable,
    *operands,
):
    """
    Drop-in replacement for `jax.lax.cond`.

    Dispatches to `jax.lax.cond` or, in branch-free mode (see
    `jaxsnn.event.set_branch_free`), evaluates both branches and selects the
    result with `jnp.where`.
    """
    if not is_branch_free():
        return jax.lax.cond(pred, true_fun, false_fun, *operands)

    true_result = true_fun(*operands)
    false_result = false_fun(*operands)
    return tree_map(
        lambda t_item, f_item: jnp.where(pred, t_item, f_item),
        true_result,
        false_result,
    )


def switch(
    index: jax.Array,
    branches: Sequence[Callable],
    *operands,
):
    """
    Drop-in replacement for `jax.lax.switch`.

    Dispatches to `jax.lax.switch` or, in branch-free mode (see
    `jaxsnn.event.set_branch_free`), evaluates every branch and selects the
    result at `index` with `jnp.where`.
    """
    if not is_branch_free():
        return jax.lax.switch(index, branches, *operands)

    all_results = [branch(*operands) for branch in branches]
    stacked = tree_map(lambda *xs: jnp.stack(xs), *all_results)
    return tree_map(lambda arr: arr[index], stacked)
