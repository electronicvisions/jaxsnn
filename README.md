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


## First Steps

Check out our examples:


```
python -m jaxsnn.event.tasks.yinyang
```

```
python -m jaxsnn.event.tasks.yinyang_event_prop
```


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
