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

```python
from jaxsnn.functional.adex import (
    adex_dynamics,
    adex_threshold_projection,
    AdexParameters,
    AdexState,
)

p = AdexParameters(
    g_l=1.0,  # nS
    Delta_T=13.0,
    tau_w_inv=1 / 22.0,
    a=30.0,  # nS
    b=0.0,
    V_l=0.0,  # mV
    V_T=0.2,  # mV
    C_m_inv=1.0,  # 1/pF
    tau_s_inv=1 / 20.0,
    v_th=0.2,  # mV
    v_reset=0.0,
)
```

```python
from jaxsnn.base import explicit
from jaxsnn.functional.threshold import triangular


def output(x, u):
    return x


def heaviside(x):
    return 0.5 + 0.5 * np.sign(x)


def gating_function(theta, threshold):
    def f(v):
        dv = threshold - v
        return heaviside(dv) * heaviside(theta - dv) * 1 / theta

    return f


dynamics = adex_dynamics(p, gating_function=gating_function(0.1, p.v_th))
equation = explicit.ExplicitConstrainedCDE(
    explicit_terms=dynamics,
    projection=adex_threshold_projection(p, triangular),
    output=output,
)
```

```python
import numpy as onp
import jaxsnn.base.funcutils as funcutils
import jax
import jax.numpy as np

initial_state = AdexState(v=p.V_l, w=0.0, s=0.0)

T = 30000
dt = 0.01
step_fn = explicit.classic_rk4_cde(equation, dt)

stim = onp.zeros(T)
stim[7000:14000] = 2.7  # add a square pulse

integrator = funcutils.controlled_trajectory(step_fn, stim)
integrator = jax.jit(integrator)
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
from ipywidgets import interact, IntSlider, FloatSlider
from functools import partial
import matplotlib.pyplot as plt

IntSlider = partial(IntSlider, continuous_update=False)
FloatSlider = partial(FloatSlider, continuous_update=True)


@interact(
    I_stim=FloatSlider(min=0.0, max=50.0, step=0.1, value=0.2),
)
def experiment(I_stim):
    stim = onp.zeros(T)
    stim[7000:8000] = I_stim  # add a square pulse
    initial_state = AdexState(v=p.V_l, w=0.0, s=0.0)
    actual = integrate_from(initial_state=initial_state, stim=stim)
    plt.plot(actual.v)
```

```python

```
