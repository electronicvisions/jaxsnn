from jaxsnn.base.types import ArrayLike

import jax.numpy as jnp
import tree_math
from typing import Callable

import dataclasses


@dataclasses.dataclass
@tree_math.struct
class AdexParameters:
    g_l: ArrayLike
    Delta_T: ArrayLike
    tau_w_inv: ArrayLike
    a: ArrayLike
    b: ArrayLike
    V_l: ArrayLike
    V_T: ArrayLike
    C_m_inv: ArrayLike
    v_th: ArrayLike
    v_reset: ArrayLike
    tau_s_inv: ArrayLike


@dataclasses.dataclass
@tree_math.struct
class AdexState:
    v: ArrayLike
    w: ArrayLike
    s: ArrayLike


def adex_dynamics(p: AdexParameters, gating_function: Callable):
    def dynamics(s: AdexState, I: ArrayLike):
        v_dot = (
            (p.g_l * p.C_m_inv) * (p.V_l - s.v)
            + (p.g_l * p.C_m_inv) * p.Delta_T * jnp.exp(1 / p.Delta_T * (s.v - p.V_T))
            - p.C_m_inv * s.w
            + p.C_m_inv * I
        )
        return AdexState(
            v=v_dot,
            w=-p.a * p.tau_w_inv * (p.V_l - s.v) - p.tau_w_inv * s.w,
            s=p.tau_s_inv * (-s.s + gating_function(s.v) * v_dot),
        )

    return dynamics


def adex_threshold_projection(p: AdexParameters, func):
    def projection(state: AdexState, _):
        # z = func(state.v)
        return AdexState(
            v=jnp.where(state.v > p.v_th, p.v_reset, state.v),
            w=jnp.where(
                state.v > p.v_th, state.w + p.b * jnp.ones_like(state.w), state.w
            ),
            s=state.s,
        )

    return projection


def adex_output(p: AdexParameters, func):
    def output(state: AdexState, _):
        z = func((state.v - p.v_th) / (p.v_th - p.v_leak))
        return z, state

    return output
