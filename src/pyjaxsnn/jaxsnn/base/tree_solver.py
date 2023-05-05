import jax.numpy as jnp
import jax.lax as lax
from functools import partial
from jaxsnn.base.types import JaxArray
import tree_math


def tree_to_matrix(d, u, p):
    """
    Auxiliary function, which turns a 'tree' matrix into
    a dense matrix. This is mostly convenient for testing.
    """
    N = d.shape[0]
    a = jnp.diag(d, 0)

    for i in range(1, N):
        a = a.at[p[i], i].set(u[i - 1])
        a = a.at[i, p[i]].set(u[i - 1])

    return a


@tree_math.struct
class TreeMatrix:
    d: JaxArray
    u: JaxArray
    p: JaxArray


@tree_math.struct
class TreeProblem:
    t: TreeMatrix
    b: JaxArray


@tree_math.struct
class TreeMatmulProblem:
    t: TreeMatrix
    b: JaxArray
    r: JaxArray


def tree_matmul(d, u, p, b):
    """
    Multiply a 'tree' matrix with a vector.

    The only non-zero entries are of the form

    a[p[i], i], a[i,i], a[i, p[i]]

    however in each row there can be many entries

    m[i] = a[i, p[i]] * b[p[i]]
    n[p[i]] = a[p[i], i] * b[i]
    diag = d * b

    """
    # TODO: Dummy implementation

    def body_fun(i, val):
        val.r = val.r.at[i].add(val.t.u[i - 1] * val.b[val.t.p[i]])
        val.r = val.r.at[val.t.p[i]].add(val.t.u[i - 1] * val.b[i])
        return val

    N = d.shape[0]

    init_val = TreeMatmulProblem(t=TreeMatrix(u=u, d=d, p=p), b=b, res=d * b)

    return lax.fori_loop(1, N, body_fun, init_val).res


def hines_solver(d, u, p, b):
    """ """
    N = d.shape[0]

    def reverse_body(j, val: TreeProblem):
        i = N - j - 1
        f = val.t.u[val.t.p[i]] / val.t.d[i]
        val.t.d = val.t.d.at[val.t.p[i]].set(
            val.t.d[val.t.p[i]] - f * val.t.u[val.t.p[i]]
        )
        val.b = val.b.at[val.t.p[i]].set(val.b[val.t.p[i]] - f * val.b[i])

        return val

    def forward_body(i, val: TreeProblem):
        val.b = val.b.at[i].set(
            (val.b[i] - val.t.u[val.t.p[i]] * val.b[val.t.p[i]]) / val.t.d[i]
        )
        return val

    problem = TreeProblem(t=TreeMatrix(d=d, u=u, p=p), b=b)
    res = lax.fori_loop(0, N - 1, reverse_body, init_val=problem)
    res.b = res.b.at[0].set(res.b[0] / res.t.d[0])
    return lax.fori_loop(1, N, forward_body, init_val=res).b


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
