import time
from functools import partial

import jax
import jax.numpy as np
import optax
from jax import random
from jaxsnn.base.compose import serial
from jaxsnn.event.dataset import yinyang_dataset
from jaxsnn.event.functional import batch_wrapper
from jaxsnn.event.leaky_integrate_and_fire import (
    LIF,
    LIFParameters,
    RecurrentEventPropLIF,
    RecurrentLIF,
)
from jaxsnn.event.loss import loss_and_acc, loss_wrapper, mse_loss
from jaxsnn.event.types import EventPropSpike
from numpy.testing import assert_almost_equal, assert_array_almost_equal


def main():
    # neuron weights, low v_reset only allows one spike per neuron
    params = LIFParameters(v_reset=-1_000.0, v_th=1.0)

    # training weights
    step_size = 5e-3
    lr_decay = 0.99
    train_samples = 5_000
    test_samples = 1_000
    batch_size = 64
    # epochs = 300
    n_train_batches = int(train_samples / batch_size)
    n_test_batches = int(test_samples / batch_size)
    t_late = 2.0 * params.tau_syn
    t_max = 4.0 * params.tau_syn
    weight_mean = [3.0, 0.5]
    weight_std = [1.6, 0.8]
    t_bias = 0.9 * params.tau_syn

    t_correct_target = 0.9 * params.tau_syn
    t_wrong_target = 1.5 * params.tau_syn

    # net
    input_size = 5
    hidden_size = 120
    output_size = 3
    n_spikes_hidden = input_size + hidden_size
    n_spikes_output = n_spikes_hidden + 3
    optimizer_fn = optax.adam

    trainset = yinyang_dataset(
        random.PRNGKey(42),
        [train_samples],
        t_late,
        mirror=True,
        t_bias=t_bias,
        t_correct_target=t_correct_target,
        t_wrong_target=t_wrong_target,
    )

    input_size = trainset[0].idx.shape[-1]

    # declare net
    init_fn_1, apply_fn_1 = serial(
        RecurrentLIF(
            layers=[hidden_size, output_size],
            n_spikes=n_spikes_output,
            t_max=t_max,
            params=params,
            mean=weight_mean,
            std=weight_std,
        )
    )

    init_fn_2, apply_fn_2 = serial(
        RecurrentEventPropLIF(
            layers=[hidden_size, output_size],
            n_spikes=n_spikes_output,
            t_max=t_max,
            params=params,
            mean=weight_mean,
            std=weight_std,
        )
    )

    init_fn_3, apply_fn_3 = serial(
        LIF(
            hidden_size,
            n_spikes=n_spikes_hidden,
            t_max=t_max,
            params=params,
            mean=weight_mean[0],
            std=weight_std[0],
        ),
        LIF(
            output_size,
            n_spikes=n_spikes_output,
            t_max=t_max,
            params=params,
            mean=weight_mean[1],
            std=weight_std[1],
        ),
    )

    rng = random.PRNGKey(42)
    weights_1 = init_fn_1(rng, input_size)
    weights_2 = init_fn_2(rng, input_size)
    weights_3 = init_fn_3(rng, input_size)

    # assert parameters equal
    assert_array_almost_equal(weights_1[0].input, weights_2[0].input)
    assert_array_almost_equal(
        weights_1[0].input[:, :hidden_size], weights_3[0].input
    )

    assert_array_almost_equal(weights_1[0].recurrent, weights_2[0].recurrent)
    assert_array_almost_equal(
        weights_1[0].recurrent[:hidden_size, hidden_size:], weights_3[1].input
    )

    sample = (
        EventPropSpike(
            trainset[0].time[0], trainset[0].idx[0], trainset[0].current[0]
        ),
        trainset[1][0],
    )
    res_1 = apply_fn_1(weights_1, sample[0])
    res_2 = apply_fn_2(weights_2, sample[0])
    res_3 = apply_fn_3(weights_3, sample[0])

    # they all produce the same output
    assert_array_almost_equal(res_1[0].time, res_2[0].time)
    assert_array_almost_equal(res_1[0].idx, res_2[0].idx)
    assert_array_almost_equal(res_1[0].current, res_2[0].current)

    assert_array_almost_equal(res_1[0].time, res_3[1].time)
    assert_array_almost_equal(res_1[0].idx, res_3[1].idx)
    assert_array_almost_equal(res_1[0].current, res_3[1].current, 4)

    # now check grads
    args = (
        mse_loss,
        params.tau_mem,
        input_size + hidden_size + output_size,
        output_size,
    )
    loss_fn_1 = partial(loss_wrapper, apply_fn_1, *args)
    loss_fn_2 = partial(loss_wrapper, apply_fn_2, *args)
    loss_fn_3 = partial(loss_wrapper, apply_fn_3, *args)

    loss_value_1, recording_1 = loss_fn_1(weights_1, sample)
    loss_value_2, recording_2 = loss_fn_2(weights_2, sample)
    loss_value_3, recording_3 = loss_fn_3(weights_3, sample)

    assert_almost_equal(loss_value_1, loss_value_2, 5)
    assert_almost_equal(loss_value_1, loss_value_3, 5)

    # check gradients
    _, grad_1 = jax.value_and_grad(loss_fn_1, has_aux=True)(weights_1, sample)
    _, grad_2 = jax.value_and_grad(loss_fn_2, has_aux=True)(weights_2, sample)
    _, grad_3 = jax.value_and_grad(loss_fn_3, has_aux=True)(weights_3, sample)

    assert_array_almost_equal(grad_1[0].input, grad_2[0].input)
    assert_array_almost_equal(grad_1[0].recurrent, grad_2[0].recurrent)

    assert_array_almost_equal(
        grad_1[0].input[:, :hidden_size], grad_3[0].input
    )
    assert_array_almost_equal(
        grad_1[0].recurrent[:hidden_size, hidden_size:], grad_3[1].input
    )

    # check gradients when no spike
    zero_weights1 = jax.tree_map(lambda p: p * 0.1, weights_1)
    zero_weights2 = jax.tree_map(lambda p: p * 0.1, weights_2)
    zerow_weights3 = jax.tree_map(lambda p: p * 0.1, weights_3)

    # check gradients
    _, grad_1 = jax.value_and_grad(loss_fn_1, has_aux=True)(
        zero_weights1, sample
    )
    _, grad_2 = jax.value_and_grad(loss_fn_2, has_aux=True)(
        zero_weights2, sample
    )
    _, grad_3 = jax.value_and_grad(loss_fn_3, has_aux=True)(
        zerow_weights3, sample
    )

    assert_array_almost_equal(grad_1[0].input, grad_2[0].input)
    assert_array_almost_equal(grad_1[0].recurrent, grad_2[0].recurrent)

    assert_array_almost_equal(
        grad_1[0].input[:, :hidden_size], grad_3[0].input
    )
    assert_array_almost_equal(
        grad_1[0].recurrent[:hidden_size, hidden_size:], grad_3[1].input
    )

    scheduler = optax.exponential_decay(step_size, train_samples, lr_decay)
    optimizer = optimizer_fn(scheduler)

    # set up optimizer
    opt_state_1 = optimizer.init(weights_1)
    opt_state_2 = optimizer.init(weights_2)
    opt_state_3 = optimizer.init(weights_3)

    def update(loss_fn, input, batch):
        opt_state, weights = input
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        value, grad = grad_fn(weights, batch)

        grad = jax.tree_util.tree_map(
            lambda par, g: np.where(par == 0.0, 0.0, g / params.tau_syn),
            weights,
            grad,
        )

        updates, opt_state = optimizer.update(grad, opt_state)
        weights = optax.apply_updates(weights, updates)
        return (opt_state, weights), (value, grad)

    update_fn_1 = partial(update, loss_fn_1)
    update_fn_2 = partial(update, loss_fn_2)
    update_fn_3 = partial(update, loss_fn_3)

    # define initial state
    state_1 = (opt_state_1, weights_1)
    state_2 = (opt_state_2, weights_2)
    state_3 = (opt_state_3, weights_3)

    # iterate through dataset
    for _ in range(1):
        state_1, _ = jax.lax.scan(update_fn_1, state_1, trainset[:2])
        state_2, _ = jax.lax.scan(update_fn_2, state_2, trainset[:2])
        state_3, _ = jax.lax.scan(update_fn_3, state_3, trainset[:2])

    # test all of them
    loss_1, acc_1, _, _ = loss_and_acc(loss_fn_1, state_1[1], trainset[:2])
    loss_2, acc_2, _, _ = loss_and_acc(loss_fn_2, state_2[1], trainset[:2])
    loss_3, acc_3, _, _ = loss_and_acc(loss_fn_3, state_3[1], trainset[:2])

    # define new dataset with batch dimension
    trainset = yinyang_dataset(
        random.PRNGKey(42),
        [n_train_batches, batch_size],
        t_late,
        mirror=True,
        t_bias=t_bias,
        t_correct_target=t_correct_target,
        t_wrong_target=t_wrong_target,
    )

    testset = yinyang_dataset(
        random.PRNGKey(1),
        [n_test_batches, batch_size],
        t_late,
        mirror=True,
        t_bias=t_bias,
        t_correct_target=t_correct_target,
        t_wrong_target=t_wrong_target,
    )

    # now add batching
    batch_loss_fn_1 = batch_wrapper(loss_fn_1)
    batch_loss_fn_2 = batch_wrapper(loss_fn_2)
    batch_loss_fn_3 = batch_wrapper(loss_fn_3)

    # try on one sample
    sample = (
        EventPropSpike(
            trainset[0].time[0], trainset[0].idx[0], trainset[0].current[0]
        ),
        trainset[1][0],
    )
    res_1, _ = batch_loss_fn_1(weights_1, sample)
    res_2, _ = batch_loss_fn_2(weights_2, sample)
    res_3, _ = batch_loss_fn_3(weights_3, sample)

    batch_update_fn_1 = partial(update, batch_loss_fn_1)
    batch_update_fn_2 = partial(update, batch_loss_fn_2)
    batch_update_fn_3 = partial(update, batch_loss_fn_3)

    # set up weights
    weights_1 = init_fn_1(rng, input_size)
    weights_2 = init_fn_2(rng, input_size)
    weights_3 = init_fn_3(rng, input_size)

    # set up optimizer
    opt_state_1 = optimizer.init(weights_1)
    opt_state_2 = optimizer.init(weights_2)
    opt_state_3 = optimizer.init(weights_3)

    # define initial state
    state_1 = (opt_state_1, weights_1)
    state_2 = (opt_state_2, weights_2)
    state_3 = (opt_state_3, weights_3)

    for i in range(5):
        start = time.time()
        state_1, _ = jax.lax.scan(batch_update_fn_1, state_1, trainset[:2])
        state_2, _ = jax.lax.scan(batch_update_fn_2, state_2, trainset[:2])
        state_3, _ = jax.lax.scan(batch_update_fn_3, state_3, trainset[:2])

    loss_1, acc_1, _, _ = loss_and_acc(
        batch_loss_fn_1, state_1[1], testset[:2]
    )
    loss_2, acc_2, _, _ = loss_and_acc(
        batch_loss_fn_2, state_2[1], testset[:2]
    )
    loss_3, acc_3, _, _ = loss_and_acc(
        batch_loss_fn_3, state_3[1], testset[:2]
    )
