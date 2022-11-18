import dataclasses
import jaxsnn.base.implicit as time_stepping
import jax.numpy as np
import tree_math

from jaxsnn.tree_solver import ArrayLike, tree_solve, tree_matmul


@tree_math.struct
class NeuronState:
    v: ArrayLike
    I: ArrayLike


@dataclasses.dataclass
class MultiCompartmentNeuronModel(time_stepping.ImplicitExplicitODE):

    d: ArrayLike
    u: ArrayLike
    p: ArrayLike

    # Parameters
    tau_syn_inv: ArrayLike

    def explicit_terms(self, state):
        return NeuronState(v=state.I, I=-self.tau_syn_inv * state.I)

    def implicit_terms(self, state):
        return NeuronState(
            v=tree_matmul(self.d, self.u, self.p, state.v), I=np.zeros_like(state.I)
        )

    def implicit_solve(self, state, step_size):
        return NeuronState(
            v=tree_solve(1 - step_size * self.d, step_size * self.u, self.p, state.v),
            I=state.I,
        )
