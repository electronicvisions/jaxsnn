"""Implicit-explicit time stepping routines for ODEs."""
import dataclasses
from typing import Callable, Sequence, TypeVar
import tree_math


PyTreeState = TypeVar("PyTreeState")
ControlInput = TypeVar("ControlInput")

TimeStepFn = Callable[[PyTreeState], PyTreeState]
ControlledTimeStepFn = Callable[[PyTreeState, ControlInput], PyTreeState]


class ImplicitExplicitODE:
    """Describes a set of ODEs with implicit & explicit terms.
    The equation is given by:
      ∂x/∂t = explicit_terms(x) + implicit_terms(x)
    `explicit_terms(x)` includes terms that should use explicit time-stepping and
    `implicit_terms(x)` includes terms that should be modeled implicitly.
    Typically the explicit terms are non-linear and the implicit terms are linear.
    This simplifies solves but isn't strictly necessary.
    """

    def explicit_terms(self, state: PyTreeState) -> PyTreeState:
        """Evaluates explicit terms in the ODE."""
        raise NotImplementedError

    def implicit_terms(self, state: PyTreeState) -> PyTreeState:
        """Evaluates implicit terms in the ODE."""
        raise NotImplementedError

    def implicit_solve(
        self,
        state: PyTreeState,
        step_size: float,
    ) -> PyTreeState:
        """Solves `y - step_size * implicit_terms(y) = x` for y."""
        raise NotImplementedError


class ImplicitExplicitCDE:
    """Describes a set of CDEs with implicit & explicit terms.

    We assume that only the explicit terms are subject to control input.

    The equation is given by:
      ∂x/∂t = explicit_terms(x, u) + implicit_terms(x)
    `explicit_terms(x, u)` includes terms that should use explicit time-stepping and are controlled
    `implicit_terms(x)` includes terms that should be modeled implicitly.
    Typically the explicit terms are non-linear and the implicit terms are linear.
    This simplifies solves but isn't strictly necessary.
    """

    def explicit_terms(self, state: PyTreeState, u: ControlInput) -> PyTreeState:
        """Evaluates explicit terms in the ODE."""
        raise NotImplementedError

    def implicit_terms(self, state: PyTreeState) -> PyTreeState:
        """Evaluates implicit terms in the ODE."""
        raise NotImplementedError

    def implicit_solve(
        self,
        state: PyTreeState,
        step_size: float,
    ) -> PyTreeState:
        """Solves `y - step_size * implicit_terms(y) = x` for y."""
        raise NotImplementedError
