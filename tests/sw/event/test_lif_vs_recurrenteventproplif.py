import unittest
import random

from functools import partial
import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jaxsnn.base.compose import serial
from jaxsnn.base.dataset import yinyang_dataset
from jaxsnn.event.modules.leaky_integrate_and_fire import (
    LIF, RecurrentEventPropLIF, LIFParameters)
from jaxsnn.event.types import EventPropSpike
from numpy.testing import (
    assert_allclose, assert_array_equal, assert_array_almost_equal)
from jaxsnn.event.encode import (
    spatio_temporal_encode,
    target_temporal_encode,
    encode
)


class TestCompareLIFVsRecurrentEventPropLIF(unittest.TestCase):

    plot_path = Path(__file__).parent.joinpath("plots")

    def setUp(self):
        self.plot_path.mkdir(exist_ok=True)

    def test_single_layer_equal(self):
        """ Note: Neurons only spike once """
        # neuron weights, low v_reset only allows one spike per neuron
        params = LIFParameters(v_reset=-1_000.0, v_th=1.0)

        # training weights
        train_samples = 5_000
        t_max = 4.0 * params.tau_syn
        weight_mean = [3.0, 0.5]
        weight_std = [1.6, 0.8]
        bias_spike = 0.9 * params.tau_syn
        t_late = 2.0 * params.tau_syn

        # net
        input_size = 5
        hidden_size = 120
        n_spikes_hidden = input_size + hidden_size

        correct_target_time = 0.9 * params.tau_syn
        wrong_target_time = 1.5 * params.tau_syn

        trainset = yinyang_dataset(
            jax.random.PRNGKey(42),
            train_samples,
            mirror=True,
            bias_spike=bias_spike
        )
        encoder_batched = jax.jit(jax.vmap(partial(
            spatio_temporal_encode,
            t_late=t_late,
            duplication=None,
            duplicate_neurons=False
        )))
        target_encoder_batched = jax.jit(jax.vmap(partial(
            target_temporal_encode,
            n_classes=3,
            correct_target_time=correct_target_time,
            wrong_target_time=wrong_target_time
        )))

        trainset = encode(
            trainset,
            encoder_batched,
            target_encoder_batched)

        input_size = trainset[0].idx.shape[-1]

        # declare net
        init_fn_1, apply_fn_1 = serial(
            LIF(
                hidden_size,
                n_spikes=n_spikes_hidden,
                t_max=t_max,
                params=params,
                mean=weight_mean[0],
                std=weight_std[0]))

        init_fn_2, apply_fn_2 = serial(
            RecurrentEventPropLIF(
                layers=[hidden_size],
                n_spikes=n_spikes_hidden,
                t_max=t_max,
                params=params,
                mean=weight_mean,
                std=weight_std,
                wrap_only_step=False))

        rng = jax.random.PRNGKey(42)
        _, weights_1 = init_fn_1(rng, input_size)
        _, weights_2 = init_fn_2(rng, input_size)

        # assert parameters equal
        self.assertIsNone(assert_array_equal(
            weights_1[0].input, weights_2[0].input))

        sample_idx = random.randint(0, train_samples)

        sample = (
            EventPropSpike(
                trainset[0].time[sample_idx],
                trainset[0].idx[sample_idx],
                trainset[0].current[sample_idx]),
            trainset[1][0])
        res_1 = apply_fn_1(weights_1, sample[0])[-1]
        res_2 = apply_fn_2(weights_2, sample[0])[-1]

        # they all produce the same output
        self.assertIsNone(assert_array_equal(
            res_1[0].time[res_1[0].idx > 4], res_2[0].time[res_2[0].idx > 4]))

        self.assertIsNone(assert_array_equal(
            res_1[0].idx[res_1[0].idx > 4], res_2[0].idx[res_2[0].idx > 4]))

        self.assertIsNone(assert_array_equal(
            res_1[0].current[res_1[0].idx > 4],
            res_2[0].current[res_2[0].idx > 4]))

        # now check grads
        def loss_fn(apply_fn, weights, batch):
            input_spikes, _ = batch
            ret = apply_fn(weights, input_spikes)
            return jnp.sum(
                jnp.where(ret[-1][0].idx >= input_size, ret[-1][0].time, 0))

        loss_fn_1 = partial(loss_fn, apply_fn_1)
        loss_fn_2 = partial(loss_fn, apply_fn_2)

        loss_value_1 = loss_fn_1(weights_1, sample)
        loss_value_2 = loss_fn_2(weights_2, sample)

        self.assertIsNone(assert_array_equal(loss_value_1, loss_value_2))

        # check gradients
        _, grad_1 = jax.value_and_grad(loss_fn_1)(weights_1, sample)
        _, grad_2 = jax.value_and_grad(loss_fn_2)(weights_2, sample)

        _, axs = plt.subplots(nrows=2)
        axs[0].plot(grad_1[0].input.reshape(-1))
        axs[0].plot(grad_2[0].input.reshape(-1))
        axs[1].plot(np.abs(grad_1[0].input - grad_2[0].input).reshape(-1))
        axs[0].set_ylabel("grad1, grad2")
        axs[1].set_ylabel("|grad1 - grad2|")
        axs[1].set_xlabel("Flat index")
        plt.savefig(
            self.plot_path.joinpath("./lif_vs_recurrenteventproplif_grads.png"))

        self.assertIsNone(assert_array_almost_equal(
            grad_1[0].input, grad_2[0].input, 4))

        # check gradients when no spike
        zero_weights1 = jax.tree_map(lambda p: p * 0.1, weights_1)
        zero_weights2 = jax.tree_map(lambda p: p * 0.1, weights_2)

        # check gradients
        _, grad_1 = jax.value_and_grad(loss_fn_1)(zero_weights1, sample)
        _, grad_2 = jax.value_and_grad(loss_fn_2)(zero_weights2, sample)

        self.assertIsNone(assert_array_almost_equal(
            grad_1[0].input, grad_2[0].input, 7))

    def test_multi_layer_equal(self):
        """ Note: Neurons only spike once """
        # neuron weights, low v_reset only allows one spike per neuron
        params = LIFParameters(v_reset=-1_000.0, v_th=1.0)

        # training weights
        train_samples = 5_000
        t_max = 4.0 * params.tau_syn
        weight_mean = [3.0, 0.5]
        weight_std = [1.6, 0.8]
        bias_spike = 0.9 * params.tau_syn
        t_late = 2.0 * params.tau_syn

        # net
        input_size = 5
        hidden_size = 120
        output_size = 3
        n_spikes_hidden = input_size + hidden_size
        n_spikes_output = n_spikes_hidden + 3

        correct_target_time = 0.9 * params.tau_syn
        wrong_target_time = 1.5 * params.tau_syn

        trainset = yinyang_dataset(
            jax.random.PRNGKey(42),
            train_samples,
            mirror=True,
            bias_spike=bias_spike
        )
        encoder_batched = jax.jit(jax.vmap(partial(
            spatio_temporal_encode,
            t_late=t_late,
            duplication=None,
            duplicate_neurons=False
        )))
        target_encoder_batched = jax.jit(jax.vmap(partial(
            target_temporal_encode,
            n_classes=3,
            correct_target_time=correct_target_time,
            wrong_target_time=wrong_target_time
        )))

        trainset = encode(
            trainset,
            encoder_batched,
            target_encoder_batched)

        input_size = trainset[0].idx.shape[-1]

        # declare net
        init_fn_1, apply_fn_1 = serial(
            LIF(
                hidden_size,
                n_spikes=n_spikes_hidden,
                t_max=t_max,
                params=params,
                mean=weight_mean[0],
                std=weight_std[0]),
            LIF(
                output_size,
                n_spikes=n_spikes_output,
                t_max=t_max,
                params=params,
                mean=weight_mean[1],
                std=weight_std[1]))

        init_fn_2, apply_fn_2 = serial(
            RecurrentEventPropLIF(
                layers=[hidden_size, output_size],
                n_spikes=n_spikes_output + 3,
                t_max=t_max,
                params=params,
                mean=weight_mean,
                std=weight_std,
                wrap_only_step=False))

        rng = jax.random.PRNGKey(42)
        _, weights_1 = init_fn_1(rng, input_size)
        _, weights_2 = init_fn_2(rng, input_size)

        # assert parameters equal
        self.assertIsNone(assert_array_equal(
            weights_1[0].input, weights_2[0].input[:, :hidden_size]))
        self.assertIsNone(assert_array_equal(
            weights_1[1].input,
            weights_2[0].recurrent[:hidden_size, hidden_size:]))

        sample_idx = random.randint(0, train_samples)

        sample = (
            EventPropSpike(
                trainset[0].time[sample_idx],
                trainset[0].idx[sample_idx],
                trainset[0].current[sample_idx]),
            trainset[1][0])
        res_1 = apply_fn_1(weights_1, sample[0])[-1]
        res_2 = apply_fn_2(weights_2, sample[0])[-1]

        self.assertIsNone(assert_array_almost_equal(
            jnp.sort(res_1[0].time[res_1[0].idx >= input_size]),
            jnp.sort(res_2[0].time[
                (res_2[0].idx >= input_size) &
                (res_2[0].idx < hidden_size + input_size)]), 4))

        self.assertIsNone(assert_array_equal(
            jnp.sort(res_1[0].idx[res_1[0].idx >= input_size]),
            jnp.sort(res_2[0].idx[
                (res_2[0].idx >= input_size) &
                (res_2[0].idx < hidden_size + input_size)])))

        self.assertIsNone(assert_array_almost_equal(
            jnp.sort(res_1[0].current[res_1[0].idx >= input_size]),
            jnp.sort(res_2[0].current[
                (res_2[0].idx >= input_size) &
                (res_2[0].idx < hidden_size + input_size)]), 1))

        self.assertIsNone(assert_array_almost_equal(
            jnp.sort(res_1[1].time[res_1[1].idx >= input_size + hidden_size]),
            jnp.sort(res_2[0].time[res_2[0].idx >= input_size + hidden_size]),
            4))

        self.assertIsNone(assert_array_equal(
            jnp.sort(res_1[1].idx[res_1[1].idx >= input_size + hidden_size]),
            jnp.sort(res_2[0].idx[res_2[0].idx >= input_size + hidden_size])))

        self.assertIsNone(assert_array_almost_equal(
            jnp.sort(res_1[1].current[res_1[1].idx >= input_size + hidden_size]),
            jnp.sort(res_2[0].current[res_2[0].idx >= input_size + hidden_size]),
            0))

        # now check grads
        def loss_fn1(apply_fn, weights, batch):
            input_spikes, _ = batch
            ret = apply_fn(weights, input_spikes)
            return jnp.sum(jnp.where(ret[-1][1].idx >= hidden_size + input_size,
                                   ret[-1][1].time, 0))

        def loss_fn2(apply_fn, weights, batch):
            input_spikes, _ = batch
            ret = apply_fn(weights, input_spikes)
            return jnp.sum(jnp.where(ret[-1][0].idx >= hidden_size + input_size,
                                   ret[-1][0].time, 0))

        loss_fn_1 = partial(loss_fn1, apply_fn_1)
        loss_fn_2 = partial(loss_fn2, apply_fn_2)

        loss_value_1 = loss_fn_1(weights_1, sample)
        loss_value_2 = loss_fn_2(weights_2, sample)

        self.assertIsNone(
            assert_array_almost_equal(loss_value_1, loss_value_2), 5)

        # check gradients
        _, grad_1 = jax.value_and_grad(loss_fn_1)(weights_1, sample)
        _, grad_2 = jax.value_and_grad(loss_fn_2)(weights_2, sample)

        _, axs = plt.subplots(nrows=4)
        axs[0].plot(grad_1[0].input.reshape(-1))
        axs[0].plot(grad_2[0].input[:, :hidden_size].reshape(-1))
        axs[1].plot(
            np.abs(grad_1[0].input
                   - grad_2[0].input[:, :hidden_size]).reshape(-1))
        axs[2].plot(grad_1[1].input.reshape(-1))
        axs[2].plot(grad_2[0].recurrent[:hidden_size, hidden_size:].reshape(-1))
        axs[3].plot(
            np.abs(grad_1[1].input
                   - grad_2[0].recurrent[:hidden_size, hidden_size:]
                   ).reshape(-1))
        axs[0].set_ylabel("grad1, grad2")
        axs[1].set_ylabel("|grad1 - grad2|")
        axs[1].set_xlabel("Index")
        axs[2].set_ylabel("grad1, grad2")
        axs[3].set_ylabel("|grad1 - grad2|")
        axs[3].set_xlabel("Index")
        plt.savefig(self.plot_path.joinpath(
            "./multi_layer_lif_vs_recurenteventproplif_grads.png"))

        self.assertIsNone(assert_array_almost_equal(
            grad_1[0].input, grad_2[0].input[:, :hidden_size], 6))
        self.assertIsNone(assert_array_almost_equal(
            grad_1[1].input, grad_2[0].recurrent[:hidden_size, hidden_size:], 6))

        # check gradients when no spike
        zero_weights1 = jax.tree_map(lambda p: p * 0.1, weights_1)
        zero_weights2 = jax.tree_map(lambda p: p * 0.1, weights_2)

        # check gradients
        _, grad_1 = jax.value_and_grad(loss_fn_1)(zero_weights1, sample)
        _, grad_2 = jax.value_and_grad(loss_fn_2)(zero_weights2, sample)

        self.assertIsNone(assert_array_equal(
            grad_1[0].input, grad_2[0].input[:, :hidden_size]))
        self.assertIsNone(assert_array_equal(
            grad_1[1].input,
            grad_2[0].recurrent[:hidden_size, hidden_size:]))

        # now check grads
        def loss_fn_batched(apply_fn, idx, weights, batch):
            input_spikes, _ = batch
            ret = jax.vmap(apply_fn, in_axes=(None, 0))(weights, input_spikes)
            return jnp.sum(jnp.where(ret[-1][idx].idx >= hidden_size + input_size,
                                   ret[-1][idx].time, 0))

        # test multiple iterations
        for i in range(10):
            sample = (
                EventPropSpike(
                    trainset[0].time[i*10: (i+1)*10],
                    trainset[0].idx[i*10: (i+1)*10],
                    trainset[0].current[i*10: (i+1)*10]),
                trainset[1][i*10: (i+1)*10])
            loss_fn_1 = partial(loss_fn_batched, apply_fn_1, 1)
            loss_fn_2 = partial(loss_fn_batched, apply_fn_2, 0)

            loss_value_1 = loss_fn_1(weights_1, sample)
            loss_value_2 = loss_fn_2(weights_2, sample)

            self.assertIsNone(
                assert_array_almost_equal(loss_value_1, loss_value_2, 5))

    def test_single_layer_multiple_spikes_equal(self):
        """ Note: Neurons spike several times """
        # neuron weights, low v_reset only allows one spike per neuron
        params = LIFParameters(v_reset=0.0, v_th=1.0)

        # training weights
        train_samples = 5_000
        t_max = 4.0 * params.tau_syn
        weight_mean = [3.0, 0.5]
        weight_std = [1.6, 0.8]
        bias_spike = 0.9 * params.tau_syn
        t_late = 2.0 * params.tau_syn

        # net
        input_size = 5
        hidden_size = 120
        n_spikes_hidden = input_size + hidden_size

        correct_target_time = 0.9 * params.tau_syn
        wrong_target_time = 1.5 * params.tau_syn

        trainset = yinyang_dataset(
            jax.random.PRNGKey(42),
            train_samples,
            mirror=True,
            bias_spike=bias_spike
        )
        encoder_batched = jax.jit(jax.vmap(partial(
            spatio_temporal_encode,
            t_late=t_late,
            duplication=None,
            duplicate_neurons=False
        )))
        target_encoder_batched = jax.jit(jax.vmap(partial(
            target_temporal_encode,
            n_classes=3,
            correct_target_time=correct_target_time,
            wrong_target_time=wrong_target_time
        )))

        trainset = encode(
            trainset,
            encoder_batched,
            target_encoder_batched)

        input_size = trainset[0].idx.shape[-1]

        # declare net
        init_fn_1, apply_fn_1 = serial(
            LIF(
                hidden_size,
                n_spikes=n_spikes_hidden,
                t_max=t_max,
                params=params,
                mean=weight_mean[0],
                std=weight_std[0]))

        init_fn_2, apply_fn_2 = serial(
            RecurrentEventPropLIF(
                layers=[hidden_size],
                n_spikes=n_spikes_hidden,
                t_max=t_max,
                params=params,
                mean=weight_mean,
                std=weight_std,
                wrap_only_step=False))

        rng = jax.random.PRNGKey(42)
        _, weights_1 = init_fn_1(rng, input_size)
        _, weights_2 = init_fn_2(rng, input_size)

        sample_idx = random.randint(0, train_samples)
        sample = (
            EventPropSpike(
                trainset[0].time[sample_idx],
                trainset[0].idx[sample_idx],
                trainset[0].current[sample_idx]),
            trainset[1][0])
        res_1 = apply_fn_1(weights_1, sample[0])[-1]
        res_2 = apply_fn_2(weights_2, sample[0])[-1]

        # Make sure we actually have more then one spike for any neuron
        _, counts = jnp.unique(res_1[0].idx, return_counts=True)
        self.assertTrue(True in (counts > 1))

        # they all produce the same output
        self.assertIsNone(assert_array_equal(
            res_1[0].time[res_1[0].idx > 4], res_2[0].time[res_2[0].idx > 4]))

        self.assertIsNone(assert_array_equal(
            res_1[0].idx[res_1[0].idx > 4], res_2[0].idx[res_2[0].idx > 4]))

        self.assertIsNone(assert_array_equal(
            res_1[0].current[res_1[0].idx > 4],
            res_2[0].current[res_2[0].idx > 4]))

        # now check grads
        def loss_fn(apply_fn, weights, batch):
            input_spikes, _ = batch
            ret = apply_fn(weights, input_spikes)
            return jnp.sum(
                jnp.where(ret[-1][0].idx >= input_size, ret[-1][0].time, 0))

        loss_fn_1 = partial(loss_fn, apply_fn_1)
        loss_fn_2 = partial(loss_fn, apply_fn_2)

        loss_value_1 = loss_fn_1(weights_1, sample)
        loss_value_2 = loss_fn_2(weights_2, sample)

        self.assertIsNone(assert_array_equal(loss_value_1, loss_value_2))

        # check gradients
        _, grad_1 = jax.value_and_grad(loss_fn_1)(weights_1, sample)
        _, grad_2 = jax.value_and_grad(loss_fn_2)(weights_2, sample)

        _, axs = plt.subplots(nrows=2)
        axs[0].plot(grad_1[0].input.reshape(-1))
        axs[0].plot(grad_2[0].input.reshape(-1))
        axs[1].plot(np.abs(grad_1[0].input - grad_2[0].input).reshape(-1))
        axs[0].set_ylabel("grad1, grad2")
        axs[1].set_ylabel("|grad1 - grad2|")
        axs[1].set_xlabel("Flat index")
        plt.savefig(
            self.plot_path.joinpath(
                "./lif_vs_recurrenteventproplif_grads_multiple_spikes.png"))

        self.assertIsNone(assert_array_almost_equal(
            grad_1[0].input, grad_2[0].input, 4))

        # check gradients when no spike
        zero_weights1 = jax.tree_map(lambda p: p * 0.1, weights_1)
        zero_weights2 = jax.tree_map(lambda p: p * 0.1, weights_2)

        # check gradients
        _, grad_1 = jax.value_and_grad(loss_fn_1)(zero_weights1, sample)
        _, grad_2 = jax.value_and_grad(loss_fn_2)(zero_weights2, sample)

        self.assertIsNone(assert_array_almost_equal(
            grad_1[0].input, grad_2[0].input, 7))

    def test_multi_layer_multiple_spikes_equal(self):
        """ Note: Neurons spike multiple times """
        # neuron weights, low v_reset only allows one spike per neuron
        params = LIFParameters(v_reset=0.0, v_th=1.0)

        # training weights
        train_samples = 5_000
        t_max = 1.0 * params.tau_syn
        weight_mean = [3.0, 0.5]
        weight_std = [1.6, 0.8]
        bias_spike = 0.9 * params.tau_syn
        t_late = 2.0 * params.tau_syn

        # net
        input_size = 5
        hidden_size = 120
        output_size = 3
        n_spikes_hidden = input_size + hidden_size
        n_spikes_output = n_spikes_hidden + 3

        correct_target_time = 0.9 * params.tau_syn
        wrong_target_time = 1.5 * params.tau_syn

        trainset = yinyang_dataset(
            jax.random.PRNGKey(42),
            train_samples,
            mirror=True,
            bias_spike=bias_spike
        )
        encoder_batched = jax.jit(jax.vmap(partial(
            spatio_temporal_encode,
            t_late=t_late,
            duplication=None,
            duplicate_neurons=False
        )))
        target_encoder_batched = jax.jit(jax.vmap(partial(
            target_temporal_encode,
            n_classes=3,
            correct_target_time=correct_target_time,
            wrong_target_time=wrong_target_time
        )))

        trainset = encode(
            trainset,
            encoder_batched,
            target_encoder_batched)

        input_size = trainset[0].idx.shape[-1]

        # declare net
        init_fn_1, apply_fn_1 = serial(
            LIF(
                hidden_size,
                n_spikes=2 * n_spikes_hidden,
                t_max=t_max,
                params=params,
                mean=weight_mean[0],
                std=weight_std[0]),
            LIF(
                output_size,
                n_spikes=2 * n_spikes_output,
                t_max=t_max,
                params=params,
                mean=weight_mean[1],
                std=weight_std[1]))

        init_fn_2, apply_fn_2 = serial(
            RecurrentEventPropLIF(
                layers=[hidden_size, output_size],
                n_spikes=2 * n_spikes_output,
                t_max=t_max,
                params=params,
                mean=weight_mean,
                std=weight_std,
                wrap_only_step=False))

        rng = jax.random.PRNGKey(42)
        _, weights_1 = init_fn_1(rng, input_size)
        _, weights_2 = init_fn_2(rng, input_size)

        # assert parameters equal
        self.assertIsNone(assert_array_equal(
            weights_1[0].input, weights_2[0].input[:, :hidden_size]))
        self.assertIsNone(assert_array_equal(
            weights_1[1].input,
            weights_2[0].recurrent[:hidden_size, hidden_size:]))

        sample_idx = random.randint(0, train_samples)
        sample = (
            EventPropSpike(
                trainset[0].time[sample_idx],
                trainset[0].idx[sample_idx],
                trainset[0].current[sample_idx]),
            trainset[1][0])
        res_1 = apply_fn_1(weights_1, sample[0])[-1]
        res_2 = apply_fn_2(weights_2, sample[0])[-1]

        # Make sure we actually have more then one spike for any neuron
        _, counts = jnp.unique(res_1[0].idx, return_counts=True)
        self.assertTrue(True in (counts > 1))

        self.assertIsNone(assert_array_almost_equal(
            jnp.sort(res_1[0].time[res_1[0].idx >= input_size]),
            jnp.sort(res_2[0].time[
                (res_2[0].idx >= input_size) &
                (res_2[0].idx < hidden_size + input_size)]), 4))

        self.assertIsNone(assert_array_equal(
            jnp.sort(res_1[0].idx[res_1[0].idx >= input_size]),
            jnp.sort(res_2[0].idx[
                (res_2[0].idx >= input_size) &
                (res_2[0].idx < hidden_size + input_size)])))

        self.assertIsNone(assert_array_almost_equal(
            jnp.sort(res_1[0].current[res_1[0].idx >= input_size]),
            jnp.sort(res_2[0].current[
                (res_2[0].idx >= input_size) &
                (res_2[0].idx < hidden_size + input_size)]), 1))

        self.assertIsNone(assert_array_almost_equal(
            jnp.sort(res_1[1].time[res_1[1].idx >= input_size + hidden_size]),
            jnp.sort(res_2[0].time[res_2[0].idx >= input_size + hidden_size]),
            4))

        self.assertIsNone(assert_array_equal(
            jnp.sort(res_1[1].idx[res_1[1].idx >= input_size + hidden_size]),
            jnp.sort(res_2[0].idx[res_2[0].idx >= input_size + hidden_size])))

        self.assertIsNone(assert_array_almost_equal(
            jnp.sort(res_1[1].current[res_1[1].idx >= input_size + hidden_size]),
            jnp.sort(res_2[0].current[res_2[0].idx >= input_size + hidden_size]),
            0))

        # now check grads
        def loss_fn1(apply_fn, weights, batch):
            input_spikes, _ = batch
            ret = apply_fn(weights, input_spikes)
            return jnp.sum(jnp.where(ret[-1][1].idx >= hidden_size + input_size,
                                   ret[-1][1].time, 0))

        def loss_fn2(apply_fn, weights, batch):
            input_spikes, _ = batch
            ret = apply_fn(weights, input_spikes)
            return jnp.sum(jnp.where(ret[-1][0].idx >= hidden_size + input_size,
                                   ret[-1][0].time, 0))

        loss_fn_1 = partial(loss_fn1, apply_fn_1)
        loss_fn_2 = partial(loss_fn2, apply_fn_2)

        loss_value_1 = loss_fn_1(weights_1, sample)
        loss_value_2 = loss_fn_2(weights_2, sample)

        self.assertIsNone(
            assert_array_almost_equal(loss_value_1, loss_value_2), 5)

        # check gradients
        _, grad_1 = jax.value_and_grad(loss_fn_1)(weights_1, sample)
        _, grad_2 = jax.value_and_grad(loss_fn_2)(weights_2, sample)

        _, axs = plt.subplots(nrows=4)
        axs[0].plot(grad_1[0].input.reshape(-1))
        axs[0].plot(grad_2[0].input[:, :hidden_size].reshape(-1))
        axs[1].plot(
            np.abs(grad_1[0].input
                   - grad_2[0].input[:, :hidden_size]).reshape(-1))
        axs[2].plot(grad_1[1].input.reshape(-1))
        axs[2].plot(grad_2[0].recurrent[:hidden_size, hidden_size:].reshape(-1))
        axs[3].plot(
            np.abs(grad_1[1].input
                   - grad_2[0].recurrent[:hidden_size, hidden_size:]
                   ).reshape(-1))
        axs[0].set_ylabel("grad1, grad2")
        axs[1].set_ylabel("|grad1 - grad2|")
        axs[1].set_xlabel("Index")
        axs[2].set_ylabel("grad1, grad2")
        axs[3].set_ylabel("|grad1 - grad2|")
        axs[3].set_xlabel("Index")
        plt.savefig(self.plot_path.joinpath(
            "./multi_layer_lif_vs_recurenteventproplif_grads_multiple_spikes.png"))

        self.assertIsNone(assert_array_almost_equal(
            grad_1[0].input, grad_2[0].input[:, :hidden_size], 6))
        self.assertIsNone(assert_array_almost_equal(
            grad_1[1].input, grad_2[0].recurrent[:hidden_size, hidden_size:],
            6))

        # check gradients when no spike
        zero_weights1 = jax.tree_map(lambda p: p * 0.1, weights_1)
        zero_weights2 = jax.tree_map(lambda p: p * 0.1, weights_2)

        # check gradients
        _, grad_1 = jax.value_and_grad(loss_fn_1)(zero_weights1, sample)
        _, grad_2 = jax.value_and_grad(loss_fn_2)(zero_weights2, sample)

        self.assertIsNone(assert_array_equal(
            grad_1[0].input, grad_2[0].input[:, :hidden_size]))
        self.assertIsNone(assert_array_equal(
            grad_1[1].input,
            grad_2[0].recurrent[:hidden_size, hidden_size:]))

        # now check grads
        def loss_fn_batched(apply_fn, idx, weights, batch):
            input_spikes, _ = batch
            ret = jax.vmap(apply_fn, in_axes=(None, 0))(weights, input_spikes)
            return jnp.sum(jnp.where(ret[-1][idx].idx >= hidden_size + input_size,
                                   ret[-1][idx].time, 0))

        # test multiple iterations
        for i in range(10):
            sample = (
                EventPropSpike(
                    trainset[0].time[i*10: (i+1)*10],
                    trainset[0].idx[i*10: (i+1)*10],
                    trainset[0].current[i*10: (i+1)*10]),
                trainset[1][i*10: (i+1)*10])
            loss_fn_1 = partial(loss_fn_batched, apply_fn_1, 1)
            loss_fn_2 = partial(loss_fn_batched, apply_fn_2, 0)

            loss_value_1 = loss_fn_1(weights_1, sample)
            loss_value_2 = loss_fn_2(weights_2, sample)

            self.assertIsNone(
                assert_allclose(loss_value_1, loss_value_2, 2e-2))


if __name__ == "__main__":
    unittest.main()