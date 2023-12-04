# jaxsnn

`jaxsnn` is an event-based approach to machine-learning-inspired training and
simulation of SNNs, including support for neuromorphic backends (BrainScaleS-2).
We build upon [jax](https://github.com/google/jax), a Python library providing
autograd and XLA functionality for high-performance machine learning research.


## Building the Software

The software builds upon existing libraries, such as
[jax](https://github.com/google/jax),
[optax](https://github.com/deepmind/optax),
and [tree-math](https://github.com/google/tree-math).
When using the neuromorphic BrainScaleS-2 backend, the software stack of the
platform is required.

We provide a container image (based on the [Singularity format](https://sylabs.io/docs/)) including all build-time and runtime dependencies.
Feel free to download the most recent version from [here](https://openproject.bioai.eu/containers/).

For all following steps, we assume that the most recent Singularity container is located at `/containers/stable/latest`.


### Github-based Build
To build this project from public resources, adhere to the following guide:

```shell
# 1) Most of the following steps will be executed within a singularity container
#    To keep the steps clutter-free, we start by defining an alias
shopt -s expand_aliases
alias c="singularity exec --app dls /containers/stable/latest"

# 2) Prepare a fresh workspace and change directory into it
mkdir workspace && cd workspace

# 3) Fetch a current copy of the symwaf2ic build tool
git clone https://github.com/electronicvisions/waf -b symwaf2ic symwaf2ic

# 4) Build symwaf2ic
c make -C symwaf2ic
ln -s symwaf2ic/waf

# 5) Setup your workspace and clone all dependencies (--clone-depth=1 to skip history)
c ./waf setup --repo-db-url=https://github.com/electronicvisions/projects --project=jaxsnn

# 6) Load PPU cross-compiler toolchain (or build https://github.com/electronicvisions/oppulance)
module load ppu-toolchain

# 7) Build the project
#    Adjust -j1 to your own needs, beware that high parallelism will increase memory consumption!
c ./waf configure
c ./waf build -j1

# 8) Install the project to ./bin and ./lib
c ./waf install

# 9) If you run programs outside waf, you'll need to add ./lib and ./bin to your path specifications
export SINGULARITYENV_PREPEND_PATH=`pwd`/bin:$SINGULARITYENV_PREPEND_PATH
export SINGULARITYENV_LD_LIBRARY_PATH=`pwd`/lib:$SINGULARITYENV_LD_LIBRARY_PATH
export PYTHONPATH=`pwd`/lib:$PYTHONPATH
```

## Structure

`jaxsnn` is split into two parts. Training of **SNNs** is done in the init/apply style.

### Time Discrete

`jaxsnn.discrete` simulates **SNNs** by treating time in a discrete way. It uses euler steps of a fixed size to advance the network forward in time which draws inspiration from [norse](www.github.com/norse/norse). 


### Time Continuous

`jaxsnn.event` treats time continously and allows jumping from one event to the next one. It's core functionality consists of the `step` function, which does three things:

1. Find the next threshold crossing
2. Integrate the neuron to this point in time
3. Apply the discontinuity after the treshold crossing

`jaxsnn.event.leaky_integrate_and_fire` provides multiple neuron types which can be used to build larger networks. Each neuron type defined the three functions mentioned above.

### BSS-2 Connection

`jaxsnn.event.hardware` provides functionality to connect to the [BSS-2 system](https://www.frontiersin.org/articles/10.3389/fnins.2022.795876/full) and to conduct learning experimens on dedicated neuromorphic hardare.


## First Steps

We provide multiple examples for usage of `jaxsnn`.

Time discrete learning using surrogate gradients on the Yin-Yang dataset:

```bash
python -m jaxsnn.discrete.tasks.yinyang
```

Event-based two layer feed-forward network with analytical gradients:

```bash
python -m jaxsnn.event.tasks.yinyang_analytical
```

Event-based recurrent network (with weights set up to emulate a two-layer feed-forward network) with gradients computed using the EventProp algorithm:

```bash
python -m jaxsnn.event.tasks.yinyang_event_prop
```

### BSS-2

If you want to work with the BSS-2 system, a working example is provided:

```bash
python -m jaxsnn.event.tasks.hardware.yinyang
```

The operation point calibration script is `src/pyjaxsnn/jaxsnn/event/hardware/calib/neuron_calib.py`.
Example:

```bash
srun -p cube --wafer 69 --fpga-without-aout 0 --pty c python ./neuron_calib.py \
	--wafer           W69F0 \
	--threshold         150 \
	--tau-syn          6e-6 \
	--tau-mem         12e-6 \
	--refractory-time 30e-6 \
	--synapse-dac-bias 1000
	--calib-dir src/pyjaxsnn/jaxsnn/event/hardware/calib
```

If you want to study the behaviour that different hardware artifacts (noise on the spike times) have on the performance of SNNs, check out this example:

```bash
python -m jaxsnn.event.tasks.hardware.yinyang_mock
```

You can switch between an actual execution on BSS-2 and a pure software mock mode, in which the hardware is emulated by a second software network. You can
add noise to spikes from this first network or limit the dynamic range (like it is on BSS-2).

## Docs

Multiple notebooks help you getting started with `jaxsnn`.

- `event_based_snn.ipynb` gives a great introduction on how to write event-based software for gradient-based learning with SNNs in JAX
- `ttfs.ipynb` explores how spikes times can be computed analytically and how a small network of LIF neurons can be constructed
- `event_prop.ipynb` compares the gradients of the EventProp algorithm analytical gradients (TTFS)

## TODO

- Numeric: In the `EventPropLIF` neuron module, gradients currently do not flow correctly over multiple layers. This problem consists because the state of the input queue is not adjusted correctly in the `custom_vjp`. It is therefore only possible to defined a multile layer networks via one recursive layer using `RecurrentEventPropLIF`
- The mapping between the hardware neuron modules `HardwareRecurrentLIF` (which can simulate multiple feed-forward layers) and the populations / projections is not yet implemented cleanly and is hacked into the tasks (experiment returns a list of spikes for two layers, which are merged together, projections are hardcoded)
- Currently, in each task and experiment, small noise is added to the spike data from hardware. This is because the `jaxsnn` gradient computation can not handle mutliple spikes with **exactly** the same time, which can happen on BSS-2 because of the cycle resolution. This should either be moved to the `experiment` class directly, or the software should be adjusted to handle this case.
- Plotting: The plotting currently does not load from saved data, but runs at the end of each task. It should be set up to run stand-alone with data loaded from a file.

## Acknowledgements

The software in this repository has been developed by staff and students
of Heidelberg University as part of the research carried out by the
Electronic Vision(s) group at the Kirchhoff-Institute for Physics.

This work has received funding from the EC Horizon 2020 Framework Programme
under grant agreements 785907 (HBP SGA2) and 945539 (HBP SGA3), the Deutsche
Forschungsgemeinschaft (DFG, German Research Foundation) under Germany's
Excellence Strategy EXC 2181/1-390900948 (the Heidelberg STRUCTURES Excellence
Cluster), the German Federal Ministry of Education and Research under grant
number 16ES1127 as part of the Pilotinnovationswettbewerb Energieeffizientes
KI-System, the Helmholtz Association Initiative and Networking Fund [Advanced
Computing Architectures (ACA)] under Project SO-092, as well as from the
Manfred Stärk Foundation, and the Lautenschläger-Forschungspreis 2018 for
Karlheinz Meier.

## Licensing

`SPDX-License-Identifier: LGPL-2.1-or-later`
