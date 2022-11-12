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

import math


def channel_time_constant(alpha, beta, v_shift=0.0, q=1.0):
    def tau(v):
        v = v + v_shift
        return 1 / q * 1 / (alpha(v) + beta(v))

    return tau


def channel_equilibrium(alpha, beta, v_shift=0.0):
    def x_inf(v):
        v = v + v_shift
        return alpha(v) / (alpha(v) + beta(v))

    return x_inf


def q_conversion(target_deg_c, original_deg_c, q10):
    return math.exp(q10, (target_deg_c - original_deg_c) / 10)
