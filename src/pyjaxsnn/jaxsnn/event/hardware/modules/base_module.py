from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional

from jaxsnn.event.hardware.observables import Observables

if TYPE_CHECKING:
    from jaxsnn.event.types import Spike


class BaseModule:

    def __init__(
        self,
        layer_idx: int
    ) -> None:
        self.layer_idx = layer_idx

    def set_params(self, params: Any) -> None:
        """ Set parameters for the module """
        pass

    def get_post_processed(self) -> Optional[Spike]:
        """ Get post-processed spikes from the module """
        return None

    @property
    def expected_return_type(self) -> Optional[Spike]:
        """ Returns the expected return type of the module """
        return None
