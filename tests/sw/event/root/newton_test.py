from functools import partial

import jax
import jax.numpy as np
from jaxsnn.base.params import LIFParameters
from jaxsnn.event.leaky_integrate import LIFState
from jaxsnn.event.root.newton import newton_solver
from numpy.testing import assert_almost_equal


def get_lif_dynamics():
    params = LIFParameters()
    kernel = np.array(
        [[-params.tau_mem_inv, params.tau_mem_inv], [0, -params.tau_syn_inv]]
    )

    def f(state, t):
        x0 = np.array([state.V, state.I])
        return np.dot(jax.scipy.linalg.expm(kernel * t), x0)

    def jc(state, t):
        return f(state, t)[0] - params.v_th

    return jc


def test_newton_solver_spike():
    solver = partial(newton_solver, get_lif_dynamics(), 0.0)

    def loss(weight):
        state = LIFState(V=0.0, I=3.0)
        state.I = state.I * weight
        return jax.jit(solver)(state, dt=0.2)

    weight = np.array(1.0)
    value, grad = jax.value_and_grad(loss)(weight)
    assert_almost_equal(value, 0.00323507, 8)
    assert_almost_equal(grad, -0.00618034, 8)


def test_newton_solver_no_spike():
    solver = partial(newton_solver, get_lif_dynamics(), 0.0)
    dt = 0.2

    def loss(state, weight):
        state.I = state.I * weight
        return jax.jit(solver)(state, dt)

    state = LIFState(V=0.0, I=2.0)
    weight = np.array(1.0)
    value, grad = jax.value_and_grad(partial(loss, state))(weight)
    assert value == dt
    assert grad == 0.0

    state = LIFState(V=0.239, I=1.368)
    value, grad = jax.value_and_grad(partial(loss, state))(weight)
    assert value == dt
    assert grad == 0.0

    state = LIFState(V=0.0, I=0.0)
    value, grad = jax.value_and_grad(partial(loss, state))(weight)
    assert value == dt
    assert grad == 0.0

    state = LIFState(V=0.0, I=-0.72)
    value, grad = jax.value_and_grad(partial(loss, state))(weight)
    assert value == dt
    assert grad == 0.0

    state = LIFState(V=0.0, I=-0.25)
    value, grad = jax.value_and_grad(partial(loss, state))(weight)
    assert value == dt
    assert grad == 0.0
