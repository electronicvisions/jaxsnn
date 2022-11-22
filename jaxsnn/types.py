from typing import Any, NamedTuple, Tuple, Union

# https://jax.readthedocs.io/en/latest/jep/12049-type-annotations.html
Array = Any


class Spike(NamedTuple):
    time: Array
    idx: Array


class StepState(NamedTuple):
    neuron_state: Array
    time: float
    running_idx: int
    input_spikes: Spike


Weight = Union[Tuple[Array, Array], Array]
