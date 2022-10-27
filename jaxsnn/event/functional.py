import jax
import jax.numpy as np


def f(A, x0, t):
    return np.dot(jax.scipy.linalg.expm(A * t), x0)  # type: ignore


def jump_condition(dynamics, v_th, x0, t):
    return dynamics(x0, t)[0] - v_th  # this implements the P y(t) - b above


def tr_equation(weights, x, spike_idx: int):
    x = x.at[spike_idx, 0].set(0.0)
    i_new = x[:, 1] + weights[spike_idx, :]
    return np.stack((x[:, 0], i_new), axis=1)


# def tr_equation(weights, x, spike_idx: int):
#     v = np.where(np.eye(weights.shape[0])[spike_idx], np.zeros_like(x[:, 0]), x[:, 0])
#     i_new = x[:, 1] + weights[spike_idx, :]
#     return np.stack((v, i_new), axis=1)


def step(dynamics, solver, tr_dynamics, weights, input_spikes, y, dt):
    t_spike = solver(y, 1e-4)

    # replace nan by inf, TODO: solver should return inf instead of nan
    t_spike = np.where(np.isnan(t_spike), np.inf, t_spike)

    # only regard future input spikes
    input_spikes = np.where(input_spikes > 0.0, input_spikes, np.inf)  # type: ignore

    # combined 'external' and 'internal' spikes
    combined = np.minimum(t_spike, input_spikes)
    spike_idx = np.argmin(combined)
    spike_time = combined[spike_idx]

    no_spike = spike_time == np.inf
    spike_idx = jax.lax.cond(no_spike, lambda: -1, lambda: spike_idx)
    t_dyn = np.minimum(spike_time, dt)
    y_minus = dynamics(y, t_dyn)

    true_fun = lambda weights, y_minus, spike_idx: y_minus
    false_fun = tr_dynamics
    args = (weights, y_minus, spike_idx)
    y_plus = jax.lax.cond(no_spike, true_fun, false_fun, *args)

    return y_plus, t_dyn, spike_idx


def forward_integration(
    step_fn,
    n_spikes,
    weights,
    input_spikes,
    t_max,
):
    def body(state, it):
        t, y = state  # t is current lower bound

        dt = t_max - t
        y_plus, dt_dyn, spike_idx = step_fn(weights, input_spikes - t, y, dt)

        t = t + dt_dyn
        state = (t, y_plus)
        return state, (t, spike_idx)

    t = 0
    return jax.lax.scan(body, (t, np.zeros((weights.shape[0], 2))), np.arange(n_spikes))
