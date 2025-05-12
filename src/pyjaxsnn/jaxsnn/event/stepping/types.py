from typing import Tuple
from jaxsnn.event.types import StepState, Weight

StepInput = Tuple[StepState, Weight, int]
