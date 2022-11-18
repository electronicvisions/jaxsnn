---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python 3.9.13 ('jaxsnn')
    language: python
    name: python3
---

# Hodgekin-Huxley Neuron

```python
import jax
import jax.numpy as np
import tree_math
import dataclasses
import matplotlib.pyplot as plt
import matplotlib as mpl
from jaxsnn.base import funcutils
from jaxsnn.base.types import ArrayLike
from jaxsnn.base import implicit
```

```python
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
```

```python

```

```python
def x0(alpha, beta):
    def x0(v):
        return alpha(v) / (alpha(v) + beta(v))

    return x0


h0 = x0(alpha_h, beta_h)
n0 = x0(alpha_n, beta_n)
m0 = x0(alpha_m, beta_m)

vs = np.arange(-100, 100, 0.1)

fig, ax = plt.subplots(1)
ax.plot(vs, h0(vs), label="h")
ax.plot(vs, n0(vs), label="n")
ax.plot(vs, m0(vs), label="m")
ax.set_xlabel("Voltage (mV)")
ax.set_ylabel("x(t)")
ax.legend()
```

```python
p = HodgekinHuxleyParameters(
    C_m_inv=1.0,  # Inverse Membrane capacitance per unit area (cm^2/uF)
    g_K=35.0,  # Average potassium channel conductance per unit area (mS/cm^2)
    g_Na=40.0,  # Average sodoum channel conductance per unit area (mS/cm^2)
    g_L=0.3,  # Average leak channel conductance per unit area (mS/cm^2)
    E_K=-77.0,  # Potassium potential (mV)
    E_Na=55.0,  # Sodium potential (mV)
    E_L=-65.0,  # Leak potential (mV)
)
```

```python
from jaxsnn.base import explicit


def projection(x, u):
    return x


def output(x, u):
    return x


dynamics = hodgekin_huxley_dynamics(p)
equation = explicit.ExplicitConstrainedCDE(
    explicit_terms=dynamics, projection=projection, output=output
)
```

```python
import numpy as onp

initial_state = HodgekinHuxleyState(v=p.E_L, m=m0(p.E_L), n=n0(p.E_L), h=h0(p.E_L))

T = 30000
dt = 0.01
step_fn = explicit.classic_rk4_cde(equation, dt)

stim = onp.zeros(T)
stim[7000:14000] = 2.7  # add a square pulse

integrator = funcutils.controlled_trajectory(step_fn, stim)
integrator = jax.jit(integrator)
```

```python
%time integrator(initial_state)[1].v.block_until_ready()  # measure JAX compilation time
%timeit integrator(initial_state)[1].v.block_until_ready()  # measure JAX runtime
```

```python
def integrate_from(initial_state, stim):
    integrator = funcutils.controlled_trajectory(step_fn, stim)
    _, actual = integrator(initial_state)
    return actual


integrate_from = jax.jit(integrate_from)
actual = integrate_from(initial_state=initial_state, stim=stim)
```

```python
def plot_hodgekin_huxley(s: HodgekinHuxleyState):
    fig, ax = plt.subplots(3, figsize=(10, 3 * 5), sharex=True)
    ts = np.arange(0, int(dt * T), dt)
    ax[0].set_ylabel("Voltage (mV)")
    ax[1].set_ylabel("Current ($\mu A$)")
    ax[2].set_ylabel("x(t)")
    ax[0].plot(ts, s.v)
    ax[1].plot(ts, stim)
    ax[2].plot(ts, s.m, label="m")
    ax[2].plot(ts, s.h, label="h")
    ax[2].plot(ts, s.n, label="n")
    ax[2].legend(loc="upper right", frameon=False)
    ax[2].set_xlabel("Time (ms)")
    return fig, ax
```

```python
from ipywidgets import interact, IntSlider, FloatSlider
from functools import partial

IntSlider = partial(IntSlider, continuous_update=False)
FloatSlider = partial(FloatSlider, continuous_update=True)


@interact(
    I_stim=FloatSlider(min=0.0, max=5.0, step=0.1, value=2.7),
)
def experiment(I_stim):
    stim = onp.zeros(T)
    stim[7000:14000] = I_stim  # add a square pulse
    initial_state = HodgekinHuxleyState(v=p.E_L, m=m0(p.E_L), n=n0(p.E_L), h=h0(p.E_L))
    actual = integrate_from(initial_state=initial_state, stim=stim)
    plot_hodgekin_huxley(actual)
```
