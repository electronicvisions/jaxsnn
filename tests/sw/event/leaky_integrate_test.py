import jax.numpy as np
from jaxsnn.base.params import LIFParameters
from jaxsnn.event.leaky_integrate import LeakyIntegrator, li_cell
from jaxsnn.event.types import Spike
from numpy.testing import assert_array_almost_equal, assert_array_equal


def test_li_cell():
    p = LIFParameters()
    t_max = 3.0 * p.tau_syn
    ts = np.linspace(0, t_max, 100)
    n_input = 3
    A = np.array([[-p.tau_mem_inv, p.tau_mem_inv], [0, -p.tau_syn_inv]])

    spikes = Spike(time=np.array([0]), idx=np.array([0]))
    weights = np.ones(n_input)
    res = li_cell(A, ts, weights, spikes)
    assert np.max(res.I) > 0.95
    assert np.argmax(res.I) == 1
    assert np.argmax(res.V) == 46


def test_leaky_integrator():
    p = LIFParameters()
    t_max = 3.0 * p.tau_syn
    n_hidden = 4
    n_input = 3
    time_steps = 100

    # test input spike previous layer
    spikes = Spike(time=np.array([1e-4]), idx=np.array([-1]))

    _, apply_fn = LeakyIntegrator(n_hidden, t_max, p, time_steps=time_steps)
    params = np.arange(n_input * n_hidden).reshape(n_input, -1)
    res = apply_fn(params, spikes)
    assert_array_equal(res.I, np.zeros((time_steps, n_hidden)))
    assert_array_equal(res.V, np.zeros((time_steps, n_hidden)))

    # test input spike
    spikes = Spike(
        time=np.array([0]),
        idx=np.array([0]),
    )
    _, apply_fn = LeakyIntegrator(n_hidden, t_max, p)
    res = apply_fn(params, spikes)
    assert_array_almost_equal(
        np.max(res.I, axis=0),
        [0.0, 0.85394, 1.707879, 2.561819],
    )
    assert_array_almost_equal(
        np.max(res.V, axis=0),
        np.array([0.0, 0.249926, 0.499852, 0.749777]),
    )

    # test inf spike time
    spikes = Spike(
        time=np.array([np.inf]),
        idx=np.array([-1]),
    )
    _, apply_fn = LeakyIntegrator(n_hidden, t_max, p)
    res = apply_fn(params, spikes)
    assert not np.any(np.isnan(res.V))
    assert not np.any(np.isnan(res.I))
