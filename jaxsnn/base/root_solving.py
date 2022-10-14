# Copyright 2022 Christian Pehle
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
import jax
import jax.numpy as jnp

def newton(f, x0):

  initial_state = (0, x0)

  def cond(state):
    it, x = state
    return (it < 10)

  def body(state):
    it, x = state
    fx, dfx = f(x), jax.grad(f)(x)
    step = fx / dfx
    new_state = it + 1, x - step
    return new_state

  return jax.lax.while_loop(
    cond,
    body,
    initial_state,
  )[1]


def bisection(f, x_min, x_max, tol):
  """Bisection root finding method

  Based on the intermediate value theorem, which
  guarantees for a continuous function that there
  is a zero in the interval [x_min, x_max] as long
  as sign(f(x_min)) != sign(f(x_max)).

  NOTE: We do not check the precondition sign(f(x_min)) != sign(f(x_max)) here
  """
  initial_state = (0, x_min, x_max)  # (iteration, x)

  def cond(state):
    it, x_min, x_max = state
    return jnp.abs(f(x_min)) > tol # it > 10 

  def body(state):
    it, x_min, x_max = state
    x = (x_min + x_max)/2

    sfxm = jnp.sign(f(x_min))
    sfx = jnp.sign(f(x)) 

    x_min = jnp.where(sfx == sfxm, x, x_min)
    x_max = jnp.where(sfx == sfxm, x_max, x)

    new_state = (it + 1, x_min, x_max)
    return new_state

  return jax.lax.while_loop(
    cond,
    body,
    initial_state,
  )[1]