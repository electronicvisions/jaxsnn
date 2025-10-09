from functools import partial

import jax
import jax.numpy as jnp
from jaxsnn.event.states import LIFState


# src of lambertw function: https://github.com/jax-ml/jax/issues/13680
def _real_lambertw_recursion(w: jax.Array, x: jax.Array) -> jax.Array:  # pylint: disable=invalid-name
    return w / (1 + w) * (1 + jnp.log(x / w))


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def _lambertwk0(x: float, max_steps: int = 5):  # pylint: disable=invalid-name
    # See https://en.wikipedia.org/wiki/Lambert_W_function#Numerical_evaluation
    w_0 = jax.lax.select(
        x > jnp.e,
        jnp.log(x) - jnp.log(jnp.log(x)),
        x / jnp.e
    )
    w_0 = jax.lax.select(
        x > 0,
        w_0,
        jnp.e * x / (1 + jnp.e * x + jnp.sqrt(1 + jnp.e * x)) * jnp.log(
            1 + jnp.sqrt(1 + jnp.e * x))
    )

    w, _ = jax.lax.scan(  # pylint: disable=invalid-name
        lambda carry, _: (_real_lambertw_recursion(carry, x),) * 2,
        w_0,
        xs=None, length=max_steps
    )

    w = jax.lax.select(  # pylint: disable=invalid-name
        jnp.isclose(x, 0.0),
        0.0,
        w
    )

    return w


@_lambertwk0.defjvp
def _lambertw_jvp(max_steps, primals, tangents):
    # Note: All branches for lambert W satisfy this JVP.
    x, = primals  # pylint: disable=invalid-name
    t, = tangents  # pylint: disable=invalid-name

    y = _lambertwk0(x, max_steps)  # pylint: disable=invalid-name
    dydx = 1 / (x + jnp.exp(y))

    jvp = jax.lax.select(
        jnp.isclose(x, -1 / jnp.e),
        jnp.nan,
        dydx * t
    )

    return y, jvp


@jnp.vectorize
def _lambertw(x, k=0, max_steps=5):  # pylint: disable=invalid-name
    if k != 0:
        raise NotImplementedError()

    return _lambertwk0(x, max_steps=max_steps)


def ttfs_ratio1_inner_most(
    a_1: jax.Array,
    b: jax.Array,  # pylint: disable=invalid-name
    w_arg: jax.Array,
    tau_mem: float,
    t_max: float,
):
    has_spike = b / a_1 > _lambertw(w_arg)
    return jax.lax.cond(
        has_spike,
        lambda: tau_mem * (b / a_1 - _lambertw(w_arg)),
        lambda: t_max,
    )


def ttfs_ratio1_inner(
    a_1: jax.Array,
    b: jax.Array,  # pylint: disable=invalid-name
    w_arg: jax.Array,
    tau_mem: float,
    t_max: float,
):
    has_spike = w_arg >= -1 / jnp.e
    return jax.lax.cond(
        has_spike,
        ttfs_ratio1_inner_most,
        lambda *args: t_max,
        a_1, b, w_arg, tau_mem, t_max
    )


def ttfs_single_time(  # pylint: disable=too-many-arguments
    tau_mem: float,
    tau_syn: float,  # pylint: disable=unused-argument
    v_th: float,
    leak: float,  # pylint: disable=unused-argument
    state: LIFState,
    t_max: float,
) -> jax.Array:
    v_0, i_0 = state.V, state.I
    a_1 = i_0
    b = -v_0  # pylint: disable=invalid-name

    w_arg = -v_th / a_1 * jnp.exp(b / a_1)
    has_spike = a_1 > 0

    return jax.lax.cond(
        has_spike,
        ttfs_ratio1_inner,
        lambda *args: t_max,
        a_1, b, w_arg, tau_mem, t_max
    )
