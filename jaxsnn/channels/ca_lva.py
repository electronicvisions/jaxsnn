# Copyright 2022, Christian Pehle
#
# LVA ca channel. Note: mtau is an approximation from the plots
# Reference: Avery and Johnston 1996, tau from Randall 1997
# shifted by -10 mv to correct for junction potential
# corrected rates using q10 = 2.3, target temperature 34, orginal 21

import jax.numpy as jnp
import tree_math

from jaxsnn.base.types import ArrayLike


@tree_math.struct
class CaLVAState:
    m: ArrayLike
    h: ArrayLike


@tree_math.struct
class CaLVAParameters:
    g_bar = 0.00001  # (S/cm2)


def g(s: CaLVAState, p: CaLVAParameters):
    return p.g_bar * s.m * s.m * s.h


def I_Ca(v: ArrayLike, e_ca: ArrayLike, s: CaLVAState, p: CaLVAParameters):
    return g(s, p) * (v - e_ca)


def m_inf(v):
    v = v + 10
    return 1.0000 / (1 + jnp.exp((v - -30.000) / -6))


def m_tau(v):
    v = v + 10
    celsius = 34
    qt = 2.3 ** ((celsius - 21) / 10)  # TODO: Find better way to adjust for temperature
    return (5.0000 + 20.0000 / (1 + jnp.exp((v - -25.000) / 5))) / qt


def h_inf(v):
    v = v + 10
    return 1.0000 / (1 + jnp.exp((v - -80.000) / 6.4))


def h_tau(v):
    v = v + 10
    celsius = 34
    qt = 2.3 ** ((celsius - 21) / 10)  # TODO: Find better way to adjust for temperature
    return (20.0000 + 50.0000 / (1 + jnp.exp((v - -40.000) / 7))) / qt


def dynamics(s: CaLVAState, v: ArrayLike):
    return CaLVAState(m=(m_inf(v) - s.m) / m_tau(v), h=(h_inf(v) - s.h) / h_tau(v))
