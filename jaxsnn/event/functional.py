from typing import Any, Callable, Tuple
from jax.experimental import checkify
import jax
import jax.numpy as np
from jax.tree_util import tree_flatten, tree_unflatten
from jax.scipy import linalg
from jaxsnn.base.types import (
    Array,
    EventPropSpike,
    InputQueue,
    Spike,
    StepState,
    Weight,
)
from jax import lax


def tree_wrap(func):
    def inner(*args):
        values, tree_def = tree_flatten(args[0])
        res = func(np.stack(values), *args[1:])
        return tree_unflatten(tree_def, res)

    return inner


def lif_wrap(func):
    def inner(*args):
        res = func(np.stack([args[0].V, args[0].I]), *args[1:])
        args[0].V = res[0]
        args[0].I = res[1]
        return args[0]

    return inner


def checkify_wrapper(f):
    def checked_fn(*args):
        error, out = checkify.checkify(
            f, errors=checkify.float_checks | checkify.user_checks
        )(*args)
        error.throw()
        return out

    def second(*args):
        err, out = checkify.checkify(checked_fn)(*args)
        return out

    return second


def batch_wrapper(func):
    def wrapped_fn(*args, **kwargs):
        res = jax.vmap(func, in_axes=(None, 0))(*args, **kwargs)
        return np.mean(res[0]), res[1]

    return wrapped_fn


def exponential_flow(A: Array):
    def flow(x0: Array, t: float):
        return np.dot(linalg.expm(A * t), x0)  # type: ignore

    return lif_wrap(flow)


def step(
    dynamics: Callable,
    solver: Callable,
    tr_dynamics: Callable,
    t_max: float,
    input: Tuple[StepState, Weight, int],
    *args: int,
) -> Tuple[Tuple[StepState, Weight, int], EventPropSpike]:
    """Determine the next spike (external or internal), and integrate the neurons to that point.
    Args:
        dynamics (Callable): Function describing neuron dynamics
        solver (Callable): Parallel root solver
        tr_dynamics (Callable): function describing the transition dynamics
        t_max (float): Max time until which to run
        weights (Weight): input and recurrent weights
        input_spikes (Spike): input spikes (time and index)
        state (StepState): (Neuron state, current_time, input_queue)
    Returns:
        Tuple[StepState, Spike]: New state after transition and spike for storing
    """
    state, weights, layer_start = input
    prev_layer_start = layer_start - weights.input.shape[0]

    pred_spikes = solver(state.neuron_state, t_max) + state.time
    spike_idx = np.argmin(pred_spikes)

    # determine spike nature and spike time
    input_time = lax.cond(
        state.input_queue.is_empty, lambda: t_max, lambda: state.input_queue.peek().time
    )
    t_dyn = np.minimum(pred_spikes[spike_idx], input_time)

    # comparing only makes sense if exactly dt is returned from solver
    spike_in_layer = pred_spikes[spike_idx] < input_time
    no_event = t_dyn + 1e-6 > t_max
    stored_idx = lax.cond(
        no_event,
        lambda: -1,
        lambda: lax.cond(
            spike_in_layer,
            lambda: spike_idx + layer_start,
            lambda: state.input_queue.peek().idx,
        ),
    )
    state = StepState(
        neuron_state=dynamics(state.neuron_state, t_dyn - state.time),
        time=t_dyn,
        input_queue=state.input_queue,
    )
    current = jax.lax.cond(
        spike_in_layer,
        lambda: state.neuron_state.I[spike_idx],
        lambda: state.input_queue.peek().current,
    )
    transitioned_state = lax.cond(
        no_event,
        lambda *args: state,
        tr_dynamics,
        state,
        weights,
        spike_idx,
        spike_in_layer,
        prev_layer_start,
    )
    return (transitioned_state, weights, layer_start), EventPropSpike(
        t_dyn, stored_idx, current
    )


def trajectory(
    dynamics: Callable, n_spikes: int
) -> Callable[[Any, Any, Any, Any], Spike]:
    def fun(initial_state, layer_start: int, weights, input_spikes) -> Spike:
        s = StepState(
            neuron_state=initial_state,
            time=0.0,
            input_queue=InputQueue(input_spikes),
        )
        _, spikes = jax.lax.scan(dynamics, (s, weights, layer_start), np.arange(n_spikes))  # type: ignore
        return spikes

    return fun
