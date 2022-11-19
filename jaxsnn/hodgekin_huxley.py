# Copyright 2022, Christian Pehle
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax.numpy as np
import tree_math

from jaxsnn.base.types import ArrayLike
import dataclasses


@dataclasses.dataclass
@tree_math.struct
class HodgekinHuxleyState:
    v: ArrayLike
    m: ArrayLike
    h: ArrayLike
    n: ArrayLike


@tree_math.struct
class HodgekinHuxleyParameters:
    g_K: ArrayLike
    E_K: ArrayLike
    g_Na: ArrayLike
    E_Na: ArrayLike
    g_L: ArrayLike
    E_L: ArrayLike
    C_m_inv: ArrayLike


def alpha_n(Vm):
    return (0.02 * (Vm - 25.0)) / (1.0 - np.exp((-1.0 * (Vm - 25.0)) / 9.0))


def beta_n(Vm):
    return (-0.002 * (Vm - 25.0)) / (1.0 - np.exp((Vm - 25.0) / 9.0))


def alpha_m(Vm):
    return (0.182 * (Vm + 35.0)) / (1.0 - np.exp((-1.0 * (Vm + 35.0)) / 9.0))


def beta_m(Vm):
    return (-0.124 * (Vm + 35.0)) / (1.0 - np.exp((Vm + 35.0) / 9.0))


def alpha_h(Vm):
    return 0.25 * np.exp((-1.0 * (Vm + 90.0)) / 12.0)


def beta_h(Vm):
    return (0.25 * np.exp((Vm + 62.0) / 6.0)) / np.exp((Vm + 90.0) / 12.0)


def I_Na(s: HodgekinHuxleyState, p: HodgekinHuxleyParameters):
    return p.g_Na * np.power(s.m, 3) * s.h * (s.v - p.E_Na)


def I_K(s: HodgekinHuxleyState, p: HodgekinHuxleyParameters):
    return p.g_K * np.power(s.n, 4) * (s.v - p.E_K)


def I_L(s: HodgekinHuxleyState, p: HodgekinHuxleyParameters):
    return p.g_L * (s.v - p.E_L)


def channel_dynamics(alpha, beta):
    def dynamics(x: ArrayLike, v: ArrayLike):
        return alpha(v) * (1 - x) - beta(v) * x

    return dynamics


def hodgekin_huxley_dynamics(p: HodgekinHuxleyParameters):
    m_dynamics = channel_dynamics(alpha_m, beta_m)
    h_dynamics = channel_dynamics(alpha_h, beta_h)
    n_dynamics = channel_dynamics(alpha_n, beta_n)

    def dynamics(s: HodgekinHuxleyState, I: ArrayLike):
        return HodgekinHuxleyState(
            v=-p.C_m_inv * (I_K(s, p) + I_Na(s, p) + I_L(s, p) - I),
            m=m_dynamics(s.m, s.v),
            n=n_dynamics(s.n, s.v),
            h=h_dynamics(s.h, s.v),
        )

    return dynamics
