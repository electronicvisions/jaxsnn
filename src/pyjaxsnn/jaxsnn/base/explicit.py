import dataclasses
from typing import Callable, Sequence, TypeVar
import jax
import tree_math


PyTreeState = TypeVar("PyTreeState")
TimeStepFn = Callable[[PyTreeState], PyTreeState]


class ExplicitConstrainedODE:
    """
    The equation is given by:
      ∂u/∂t = explicit_terms(u)
      0 = constraint(u)
    """

    def __init__(self, explicit_terms, projection):
        self.explicit_terms = explicit_terms
        self.projection = projection

    def explicit_terms(self, state):
        """Explicitly evaluate the ODE."""
        raise NotImplementedError

    def projection(self, state):
        """Enforce the constraint."""
        raise NotImplementedError


class ExplicitConstrainedCDE:
    """

    The equation is given by:
      ∂u/∂t = explicit_terms(u, x)
      0 = constraint(u)
    """

    def __init__(self, explicit_terms, projection, output):
        self.explicit_terms = explicit_terms
        self.projection = projection
        self.output = output

    def explicit_terms(self, state, input):
        """Explicitly evaluate the ODE."""
        raise NotImplementedError

    def projection(self, state, input):
        """Enforce the constraint."""
        raise NotImplementedError

    def output(self, state):
        """"""
        raise NotImplementedError
