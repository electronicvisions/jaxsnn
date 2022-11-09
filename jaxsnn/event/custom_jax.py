import jax
from jax.tree_util import tree_flatten, tree_map, tree_unflatten


def scan(f, init, xs, length=None):
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
