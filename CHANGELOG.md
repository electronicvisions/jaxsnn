# CHANGELOG

This document tracks user-facing API changes.

## v0.3.0 — New jaxsnn API

This version introduces a **breaking change** for describing network topologies for both time-stepped and event-based simulation using a simple add/connect/done builder pattern.
The BSS-2 hardware backend is now properly integrated via the new pygrenade API.

See the example below.

### Relevant changes:

* Topology builder: unified add/connect/done interface, shared by the event-based and discrete (time-stepped) namespaces
* Event-based topology supports selectable backprop (EventProp or analytical) and hardware/mock execution
* Module refactor: split into functional (dynamics/transitions), adjoint (backward dynamics/transitions), and modules (composed functionality like LIF), with an hx sub-namespace for BSS-2 modules
  * BSS-2 specific code (pygrenade populations, projections, experiment description) lives in `jaxsnn/event/hardware`
* Renamed/moved:
  * new solver namespace, ttfs → lif_analytical
  * NIR utilities moved into `event/utils/`
  * Dropped jaxsnn mock legacy code
* New `yinyang_bss2.py` example for BSS-2 hardware execution
* New topology, module-correctness, and analytical-vs-EventProp gradient parity tests

### API Example

```python
jaxsnn.init_hardware()

# define topology
builder = Topology(
    mock=False,
    t_max=t_max,
    backprop_method="eventprop",
)

# create modules
builder.add(
    {
        "inp": HXSource(size=input_size, n_events=n_events_in),
        "lif_h": HXLIF(size=size, n_steps=n_steps, n_hw_spikes=n_spikes, params=lif_params),
        "lif_o": HXLIF(size=...),
        "syn_ih": HXLinear(mean=0.6, std=0.3, weight_scale=weight_scale),
        "syn_ho": HXLinear(...),
    }
)

# connect modules
builder.connect(
    [
        ("inp", "syn_ih"),
        ("syn_ih", "lif_h"),
        ("lif_h", "syn_ho"),
        ("syn_ho", "lif_o"),
    ]
)

init_fn, apply_fn = builder.done()
params = init_fn(param_rng)
result = apply_fn(inputs)

jaxsnn.release_hardware()
```

More examples live in `jaxsnn.examples.{event, discrete}`.
