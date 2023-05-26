import jax
from jax.tree_util import tree_flatten, tree_map, tree_unflatten


def scan(f, init, xs, length=None, reverse=False, print_bool=True):
    xs_flat, xs_tree = tree_flatten(xs)
    carry = init
    ys = []
    length = len(xs_flat[0])
    for i in range(length):
        if print_bool:
            print(f"Sample {i}")
        if reverse:
            i = length - i - 1
        xs_slice = [x[i] for x in xs_flat]
        carry, y = f(carry, tree_unflatten(xs_tree, xs_slice))
        ys.append(y)
    stack = lambda *ys: jax.numpy.stack(ys)
    stacked_y = tree_map(stack, *ys)
    return carry, stacked_y


def simple_scan(f, init, xs, length=None):
    xs_flat, xs_tree = tree_flatten(xs)
    carry = init
    ys = []
    length = len(xs_flat[0])
    for i in range(length):
        xs_slice = [x[i] for x in xs_flat]
        carry, y = f(carry, tree_unflatten(xs_tree, xs_slice))
        ys.append(y)
    stack = lambda *ys: jax.numpy.stack(ys)
    stacked_y = tree_map(stack, *ys)
    return carry, stacked_y


def cond(pred, true_fun, false_fun, *operands):
    """Call both function to evaluate compiled behaviour of `jax.lax.scan`"""
    true_result = true_fun(*operands)
    false_result = false_fun(*operands)
    return jax.tree_map(
        lambda t, f: jax.lax.select(pred, t, f), true_result, false_result
    )
