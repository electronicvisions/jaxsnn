import jax

from hxtorch.core.modules.projection import Projection as CoreProjection

from jaxsnn.event.hardware.modules.base_module import BaseModule


class Projection(CoreProjection, BaseModule):

    def __init__(
        self,
        layer_idx: int,
        *args,
        **kwargs
    ) -> None:
        CoreProjection.__init__(self, *args, **kwargs)
        BaseModule.__init__(self, layer_idx)

    def set_params(self, weight: jax.Array) -> None:
        self.weight = weight
