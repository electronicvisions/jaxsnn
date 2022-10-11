import dataclasses
from typing import Callable, Optional
import jaxsnn.base.implicit as time_stepping


@dataclasses.dataclass
class MultiCompartmentNeuronModel(time_stepping.ImplicitExplicitODE):
  """
  """
  def __post_init__(self):
    pass

  def explicit_terms(self, u):
    """Non-linear parts of the equation, """
    pass

  def implicit_terms(self, u):
    """Linear parts of the equation, """
    pass

  def implicit_solve(self, uhat, time_step):
    """Solves for `implicit_terms`, implicitly."""
    pass 