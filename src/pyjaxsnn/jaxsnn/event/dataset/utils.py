from typing import Tuple

import jax
import jax.numpy as np
from jaxsnn.event.types import EventPropSpike, Spike

Dataset = Tuple[EventPropSpike, jax.Array, str]


def add_current(spike: Spike) -> EventPropSpike:
    return EventPropSpike(
        spike.time, spike.idx, np.zeros_like(spike.idx, dtype=spike.time.dtype)
    )
