import tree_math

from jaxsnn.base.types import ArrayLike


@tree_math.struct
class NeuronState:
    v: ArrayLike
    I: ArrayLike
