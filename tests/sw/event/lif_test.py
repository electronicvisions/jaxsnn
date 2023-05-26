from functools import partial
import time
import jax
import jax.numpy as np
import optax
from jax import random
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from jaxsnn.base.types import EventPropSpike
from jaxsnn.event.compose import serial
from jaxsnn.event.dataset import yinyang_dataset
from jaxsnn.event.functional import batch_wrapper
from jaxsnn.event.leaky_integrate_and_fire import (
    LIF,
    LIFParameters,
    RecurrentEventPropLIF,
    RecurrentLIF,
)
from jaxsnn.event.loss import loss_and_acc, loss_wrapper, mse_loss
from jaxsnn.event.root import ttfs_solver
from jax.config import config

config.update("jax_debug_nans", True)


def main():
    # neuron params, low v_reset only allows one spike per neuron
    p = LIFParameters(v_reset=-1_000.0, v_th=1.0)

    # training params
    step_size = 5e-3
    lr_decay = 0.99
    train_samples = 5_000
    test_samples = 1_000
    batch_size = 64
    # epochs = 300
    n_train_batches = int(train_samples / batch_size)
    n_test_batches = int(test_samples / batch_size)
    t_late = 2.0 * p.tau_syn
    t_max = 4.0 * p.tau_syn
    weight_mean = [3.0, 0.5]
    weight_std = [1.6, 0.8]
    bias_spike = 0.9 * p.tau_syn

    correct_target_time = 0.9 * p.tau_syn
    wrong_target_time = 1.5 * p.tau_syn

    # net
    input_size = 5
    hidden_size = 120
    output_size = 3
    n_spikes_hidden = input_size + hidden_size
    n_spikes_output = n_spikes_hidden + 3
    optimizer_fn = optax.adam

    trainset = yinyang_dataset(
        random.PRNGKey(42),
        t_late,
        [train_samples],
        mirror=True,
        bias_spike=bias_spike,
        correct_target_time=correct_target_time,
        wrong_target_time=wrong_target_time,
    )

    input_size = trainset[0].idx.shape[-1]
    solver = partial(ttfs_solver, p.tau_mem, p.v_th)

    # declare net
    init_fn_1, apply_fn_1 = serial(
        RecurrentLIF(
            layers=[hidden_size, output_size],
            n_spikes=n_spikes_output,
            t_max=t_max,
            p=p,
            solver=solver,
            mean=weight_mean,
            std=weight_std,
        )
    )

    init_fn_2, apply_fn_2 = serial(
        RecurrentEventPropLIF(
            layers=[hidden_size, output_size],
            n_spikes=n_spikes_output,
            t_max=t_max,
            p=p,
            solver=solver,
            mean=weight_mean,
            std=weight_std,
        )
    )

    init_fn_3, apply_fn_3 = serial(
        LIF(
            hidden_size,
            n_spikes=n_spikes_hidden,
            t_max=t_max,
            p=p,
            solver=solver,
            mean=weight_mean[0],
            std=weight_std[0],
        ),
        LIF(
            output_size,
            n_spikes=n_spikes_output,
            t_max=t_max,
            p=p,
            solver=solver,
            mean=weight_mean[1],
            std=weight_std[1],
        ),
    )

    rng = random.PRNGKey(42)
    params_1 = init_fn_1(rng, input_size)
    params_2 = init_fn_2(rng, input_size)
    params_3 = init_fn_3(rng, input_size)

    # assert parameters equal
    assert_array_almost_equal(params_1[0].input, params_2[0].input)
    assert_array_almost_equal(params_1[0].input[:, :hidden_size], params_3[0].input)

    assert_array_almost_equal(params_1[0].recurrent, params_2[0].recurrent)
    assert_array_almost_equal(
        params_1[0].recurrent[:hidden_size, hidden_size:], params_3[1].input
    )

    sample = (
        EventPropSpike(trainset[0].time[0], trainset[0].idx[0], trainset[0].current[0]),
        trainset[1][0],
    )
    res_1 = apply_fn_1(params_1, sample[0])
    res_2 = apply_fn_2(params_2, sample[0])
    res_3 = apply_fn_3(params_3, sample[0])

    # they all produce the same output
    assert_array_almost_equal(res_1[0].time, res_2[0].time)
    assert_array_almost_equal(res_1[0].idx, res_2[0].idx)
    assert_array_almost_equal(res_1[0].current, res_2[0].current)

    assert_array_almost_equal(res_1[0].time, res_3[1].time)
    assert_array_almost_equal(res_1[0].idx, res_3[1].idx)
    assert_array_almost_equal(res_1[0].current, res_3[1].current, 4)

    # now check grads
    args = (mse_loss, p.tau_mem, input_size + hidden_size + output_size, output_size)
    loss_fn_1 = partial(loss_wrapper, apply_fn_1, *args)
    loss_fn_2 = partial(loss_wrapper, apply_fn_2, *args)
    loss_fn_3 = partial(loss_wrapper, apply_fn_3, *args)

    loss_value_1, recording_1 = loss_fn_1(params_1, sample)
    loss_value_2, recording_2 = loss_fn_2(params_2, sample)
    loss_value_3, recording_3 = loss_fn_3(params_3, sample)

    assert_almost_equal(loss_value_1, loss_value_2, 5)
    assert_almost_equal(loss_value_1, loss_value_3, 5)

    # check gradients
    _, grad_1 = jax.value_and_grad(loss_fn_1, has_aux=True)(params_1, sample)
    _, grad_2 = jax.value_and_grad(loss_fn_2, has_aux=True)(params_2, sample)
    _, grad_3 = jax.value_and_grad(loss_fn_3, has_aux=True)(params_3, sample)

    assert_array_almost_equal(grad_1[0].input, grad_2[0].input)
    assert_array_almost_equal(grad_1[0].recurrent, grad_2[0].recurrent)

    assert_array_almost_equal(grad_1[0].input[:, :hidden_size], grad_3[0].input)
    assert_array_almost_equal(
        grad_1[0].recurrent[:hidden_size, hidden_size:], grad_3[1].input
    )

    # check gradients when no spike
    zero_params1 = jax.tree_map(lambda p: p * 0.1, params_1)
    zero_params2 = jax.tree_map(lambda p: p * 0.1, params_2)
    zero_params3 = jax.tree_map(lambda p: p * 0.1, params_3)

    # check gradients
    _, grad_1 = jax.value_and_grad(loss_fn_1, has_aux=True)(zero_params1, sample)
    _, grad_2 = jax.value_and_grad(loss_fn_2, has_aux=True)(zero_params2, sample)
    _, grad_3 = jax.value_and_grad(loss_fn_3, has_aux=True)(zero_params3, sample)

    assert_array_almost_equal(grad_1[0].input, grad_2[0].input)
    assert_array_almost_equal(grad_1[0].recurrent, grad_2[0].recurrent)

    assert_array_almost_equal(grad_1[0].input[:, :hidden_size], grad_3[0].input)
    assert_array_almost_equal(
        grad_1[0].recurrent[:hidden_size, hidden_size:], grad_3[1].input
    )

    scheduler = optax.exponential_decay(step_size, train_samples, lr_decay)
    optimizer = optimizer_fn(scheduler)

    # set up optimizer
    opt_state_1 = optimizer.init(params_1)
    opt_state_2 = optimizer.init(params_2)
    opt_state_3 = optimizer.init(params_3)

    def update(loss_fn, input, batch):
        opt_state, params = input
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        value, grad = grad_fn(params, batch)

        grad = jax.tree_util.tree_map(
            lambda par, g: np.where(par == 0.0, 0.0, g / p.tau_syn), params, grad
        )

        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        return (opt_state, params), (value, grad)

    update_fn_1 = partial(update, loss_fn_1)
    update_fn_2 = partial(update, loss_fn_2)
    update_fn_3 = partial(update, loss_fn_3)

    # define initial state
    state_1 = (opt_state_1, params_1)
    state_2 = (opt_state_2, params_2)
    state_3 = (opt_state_3, params_3)

    # iterate through dataset
    for _ in range(1):
        state_1, _ = jax.lax.scan(update_fn_1, state_1, trainset[:2])
        state_2, _ = jax.lax.scan(update_fn_2, state_2, trainset[:2])
        state_3, _ = jax.lax.scan(update_fn_3, state_3, trainset[:2])

    # test all of them
    loss_1, acc_1, _, _ = loss_and_acc(loss_fn_1, state_1[1], trainset[:2])
    loss_2, acc_2, _, _ = loss_and_acc(loss_fn_2, state_2[1], trainset[:2])
    loss_3, acc_3, _, _ = loss_and_acc(loss_fn_3, state_3[1], trainset[:2])

    print(f"Achieved accuracy: {acc_1:.3f}, {acc_2:.3f}, {acc_3:.3f}")

    # define new dataset with batch dimension
    trainset = yinyang_dataset(
        random.PRNGKey(42),
        t_late,
        [n_train_batches, batch_size],
        mirror=True,
        bias_spike=bias_spike,
        correct_target_time=correct_target_time,
        wrong_target_time=wrong_target_time,
    )

    testset = yinyang_dataset(
        random.PRNGKey(1),
        t_late,
        [n_test_batches, batch_size],
        mirror=True,
        bias_spike=bias_spike,
        correct_target_time=correct_target_time,
        wrong_target_time=wrong_target_time,
    )

    # now add batching
    batch_loss_fn_1 = batch_wrapper(loss_fn_1)
    batch_loss_fn_2 = batch_wrapper(loss_fn_2)
    batch_loss_fn_3 = batch_wrapper(loss_fn_3)

    # try on one sample
    sample = (
        EventPropSpike(trainset[0].time[0], trainset[0].idx[0], trainset[0].current[0]),
        trainset[1][0],
    )
    res_1, _ = batch_loss_fn_1(params_1, sample)
    res_2, _ = batch_loss_fn_2(params_2, sample)
    res_3, _ = batch_loss_fn_3(params_3, sample)
    print(f"Batched loss: {res_1:.5f}, {res_2:.5f}, {res_3:.5f}")

    batch_update_fn_1 = partial(update, batch_loss_fn_1)
    batch_update_fn_2 = partial(update, batch_loss_fn_2)
    batch_update_fn_3 = partial(update, batch_loss_fn_3)

    # set up params
    params_1 = init_fn_1(rng, input_size)
    params_2 = init_fn_2(rng, input_size)
    params_3 = init_fn_3(rng, input_size)

    # set up optimizer
    opt_state_1 = optimizer.init(params_1)
    opt_state_2 = optimizer.init(params_2)
    opt_state_3 = optimizer.init(params_3)

    # define initial state
    state_1 = (opt_state_1, params_1)
    state_2 = (opt_state_2, params_2)
    state_3 = (opt_state_3, params_3)

    for i in range(5):
        start = time.time()
        state_1, _ = jax.lax.scan(batch_update_fn_1, state_1, trainset[:2])
        state_2, _ = jax.lax.scan(batch_update_fn_2, state_2, trainset[:2])
        state_3, _ = jax.lax.scan(batch_update_fn_3, state_3, trainset[:2])
        print(f"Finished epoch {i} in {time.time()-start:.2f} seconds")

    loss_1, acc_1, _, _ = loss_and_acc(batch_loss_fn_1, state_1[1], testset[:2])
    loss_2, acc_2, _, _ = loss_and_acc(batch_loss_fn_2, state_2[1], testset[:2])
    loss_3, acc_3, _, _ = loss_and_acc(batch_loss_fn_3, state_3[1], testset[:2])
    print(
        f"Achieved accuracy with badge size {batch_size}: {acc_1:.3f}, {acc_2:.3f}, {acc_3:.3f}"
    )


if __name__ == "__main__":
    main()
