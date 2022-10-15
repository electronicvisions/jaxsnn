from jaxsnn.tree_solver import ArrayLike

import jax.numpy as jnp
import tree_math

@tree_math.struct
class AdexParameters:
    g_l : ArrayLike
    g_exp : ArrayLike
    Delta_exp : ArrayLike
    tau_w_inv : ArrayLike
    a : ArrayLike
    V_l : ArrayLike
    V_exp : ArrayLike
    C_m_inv : ArrayLike
    tau_s_inv : ArrayLike
    v_th : ArrayLike
    v_reset : ArrayLike

@tree_math.struct
class AdexState:
    v : ArrayLike
    w : ArrayLike
    s : ArrayLike

def adex_dynamics(p: AdexParameters, gating_function: Callable):
    def dynamics(s : AdexState):
        v_dot = (p.g_l * p.C_m_inv) * (p.V_l - s.v) + (p.g_exp * p.C_m_inv) * p.Delta_exp * jnp.exp(1/p.Delta_exp * (s.v - p.V_exp)) - p.C_m_inv * s.w
        return AdexState(
            v = v_dot,
            w = -p.a * p.tau_w_inv * (p.V_l - s.v) - p.tau_w_inv * s.w,
            s = p.tau_s_inv * (-s.s + gating_function(s.v) * v_dot),
        )
    return dynamics

def adex_threshold_projection(p : AdexParameters):
    def projection(state : AdexState):
        return AdexState(
            v = jnp.where(state.v > p.v_th, p.v_reset, state.v),
            w = jnp.where(state.v > p.v_th, jnp.ones_like(state.w), state.w),
            s = state.s
        )
    return projection