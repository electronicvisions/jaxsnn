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

# Point Neuron Models


## LIF Neuron

```python
import jaxsnn.base.explicit as explicit
import jaxsnn.base.funcutils as funcutils
import jaxsnn.functional.threshold as threshold
import jaxsnn.functional.leaky_integrate_and_fire as lif

import jax
import numpy as np
```

### Leak over Threshold

```python
p = lif.LIFParameters(v_leak=0.6, v_th=0.5)
initial_state = lif.LIFState(v=p.v_reset, I=0.0, w_rec=0.0)

T = 1000
dt = 0.0001
step_fn = explicit.classic_rk4_cde(lif.lif_equation(p, threshold.triangular), dt)

stim = lif.LIFInput(I=np.zeros(T), z=np.zeros(T))
integrator = funcutils.controlled_trajectory(step_fn, stim)
integrator = jax.jit(integrator)
```

```python
_, (spikes, state) = integrator(initial_state)
```

```python
import matplotlib.pyplot as plt

# plt.plot(state.I)
plt.plot(state.v)
```
