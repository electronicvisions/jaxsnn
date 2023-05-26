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


@dataclasses.dataclass
class ButcherTableau:
    a: Sequence[Sequence[float]]
    b: Sequence[float]
    # TODO(shoyer): add c, when we support time-dependent equations.

    def __post_init__(self):
        if len(self.a) + 1 != len(self.b):
            raise ValueError("inconsistent Butcher tableau")


def constrained_rk(
    tableau: ButcherTableau,
    equation: ExplicitConstrainedODE,
    time_step: float,
) -> TimeStepFn:
    """Create a forward Runge-Kutta time-stepper for incompressible Navier-Stokes.
    This function implements the reference method (equations 16-21), rather than
    the fast projection method, from:
      "Fast-Projection Methods for the Incompressible Navier–Stokes Equations"
      Fluids 2020, 5, 222; doi:10.3390/fluids5040222
    Args:
      tableau: Butcher tableau.
      equation: equation to use.
      time_step: overall time-step size.
    Returns:
      Function that advances one time-step forward.
    """
    # pylint: disable=invalid-name
    dt = time_step
    F = tree_math.unwrap(equation.explicit_terms)
    P = tree_math.unwrap(equation.projection)

    a = tableau.a
    b = tableau.b
    num_steps = len(b)

    @tree_math.wrap
    def step_fn(u0):
        u = [None] * num_steps
        k = [None] * num_steps

        u[0] = u0
        k[0] = F(u0)

        for i in range(1, num_steps):
            u_star = u0 + dt * sum(a[i - 1][j] * k[j] for j in range(i) if a[i - 1][j])
            u[i] = P(u_star)
            k[i] = F(u[i])

        u_star = u0 + dt * sum(b[j] * k[j] for j in range(num_steps) if b[j])
        u_final = P(u_star)

        return u_final

    return step_fn


def constrained_cde_rk(
    tableau: ButcherTableau,
    equation: ExplicitConstrainedCDE,
    time_step: float,
) -> TimeStepFn:
    """Create a forward Runge-Kutta time-stepper for incompressible Navier-Stokes.
    This function implements the reference method (equations 16-21), rather than
    the fast projection method, from:
      "Fast-Projection Methods for the Incompressible Navier–Stokes Equations"
      Fluids 2020, 5, 222; doi:10.3390/fluids5040222
    Args:
      tableau: Butcher tableau.
      equation: equation to use.
      time_step: overall time-step size.
    Returns:
      Function that advances one time-step forward.
    """
    # pylint: disable=invalid-name
    dt = time_step
    F = tree_math.unwrap(equation.explicit_terms)
    P = tree_math.unwrap(equation.projection)
    O = tree_math.unwrap(equation.output)

    a = tableau.a
    b = tableau.b
    num_steps = len(b)

    @tree_math.wrap
    def step_fn(u0, x0):
        u = [None] * num_steps
        k = [None] * num_steps

        u[0] = u0
        k[0] = F(u0, x0)

        for i in range(1, num_steps):
            u_star = u0 + dt * sum(a[i - 1][j] * k[j] for j in range(i) if a[i - 1][j])
            u[i] = P(u_star, x0)
            k[i] = F(u[i], x0)

        u_star = u0 + dt * sum(b[j] * k[j] for j in range(num_steps) if b[j])

        out = O(u_star, x0)
        u_final = P(u_star, x0)

        return u_final, out

    return step_fn


def forward_euler(
    equation: ExplicitConstrainedODE,
    time_step: float,
) -> TimeStepFn:
    return jax.named_call(
        constrained_rk(ButcherTableau(a=[], b=[1]), equation, time_step),
        name="forward_euler",
    )


def forward_euler_cde(
    equation: ExplicitConstrainedCDE,
    time_step: float,
) -> TimeStepFn:
    return jax.named_call(
        constrained_cde_rk(ButcherTableau(a=[], b=[1]), equation, time_step),
        name="forward_euler",
    )


def midpoint_rk2(
    equation: ExplicitConstrainedODE,
    time_step: float,
) -> TimeStepFn:
    return jax.named_call(
        constrained_rk(
            ButcherTableau(a=[[1 / 2]], b=[0, 1]),
            equation=equation,
            time_step=time_step,
        ),
        name="midpoint_rk2",
    )


def midpoint_rk2_cde(
    equation: ExplicitConstrainedCDE,
    time_step: float,
) -> TimeStepFn:
    return jax.named_call(
        constrained_cde_rk(
            ButcherTableau(a=[[1 / 2]], b=[0, 1]),
            equation=equation,
            time_step=time_step,
        ),
        name="midpoint_rk2",
    )


def heun_rk2(
    equation: ExplicitConstrainedODE,
    time_step: float,
) -> TimeStepFn:
    return jax.named_call(
        constrained_rk(
            ButcherTableau(a=[[1]], b=[1 / 2, 1 / 2]),
            equation=equation,
            time_step=time_step,
        ),
        name="heun_rk2",
    )


def heun_rk2_cde(
    equation: ExplicitConstrainedCDE,
    time_step: float,
) -> TimeStepFn:
    return jax.named_call(
        constrained_cde_rk(
            ButcherTableau(a=[[1]], b=[1 / 2, 1 / 2]),
            equation=equation,
            time_step=time_step,
        ),
        name="heun_rk2",
    )


def classic_rk4(
    equation: ExplicitConstrainedODE,
    time_step: float,
) -> TimeStepFn:
    return jax.named_call(
        constrained_rk(
            ButcherTableau(
                a=[[1 / 2], [0, 1 / 2], [0, 0, 1]], b=[1 / 6, 1 / 3, 1 / 3, 1 / 6]
            ),
            equation=equation,
            time_step=time_step,
        ),
        name="classic_rk4",
    )


def classic_rk4_cde(
    equation: ExplicitConstrainedCDE,
    time_step: float,
) -> TimeStepFn:
    return jax.named_call(
        constrained_cde_rk(
            ButcherTableau(
                a=[[1 / 2], [0, 1 / 2], [0, 0, 1]], b=[1 / 6, 1 / 3, 1 / 3, 1 / 6]
            ),
            equation=equation,
            time_step=time_step,
        ),
        name="classic_rk4",
    )
