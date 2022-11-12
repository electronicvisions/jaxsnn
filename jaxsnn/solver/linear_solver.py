from re import A
import jax.numpy as np
import jax.lax as lax
from functools import partial


def tree_to_matrix(d, u, p):
    """
    Auxiliary function, which turns a 'tree' matrix into
    a dense matrix. This is mostly convenient for testing.
    """
    N = d.shape[0]
    a = np.diag(d, 0)

    for i in range(1, N):
        a = a.at[p[i], i].set(u[i - 1])
        a = a.at[i, p[i]].set(u[i - 1])
    return a


def tree_matmul(d, u, p, b):
    """
    Multiply a 'tree' matrix with a vector.
    """
    # TODO: Dummy implementation
    m = tree_to_matrix(d, u, p)
    return np.dot(m, b)


def hines_solver(d, u, p, b):
    """ """
    N = d.shape[0]

    for i in range(N - 1, 0, -1):
        f = u[p[i]] / d[i]
        d = d.at[p[i]].set(d[p[i]] - f * u[p[i]])
        b = b.at[p[i]].set(b[p[i]] - f * b[i])

    b = b.at[0].set(b[0] / d[0])

    for i in range(1, N):
        b = b.at[i].set((b[i] - u[p[i]] * b[p[i]]) / d[i])

    return b


def tree_solve(d, u, p, b):
    """
    A solver for 'tree' matrices, which is compatible with the jax tracer.
    """
    solver = partial(
        lax.custom_linear_solve,
        lambda x: tree_matmul(d, u, p, x),
        solve=lambda _, x: hines_solver(d, u, p, x),
        symmetric=True,
    )

    return solver(b)
