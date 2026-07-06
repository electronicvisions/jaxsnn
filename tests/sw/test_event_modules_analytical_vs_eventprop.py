import unittest
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

import jax
import jax.numpy as jnp

from jaxsnn.event.types import Spike
from jaxsnn.event.topology import Topology
from jaxsnn.event.modules import (
    LIF,
    Linear,
    Source,
    LIFParameters,
)

from numpy.testing import (
    assert_array_equal,
    assert_array_almost_equal,
)

try:
    from jax.tree import map as tree_map
except ImportError:
    # for compatibility with jax@:0.4.25
    from jax.tree_util import tree_map


class TestCompareLIFVsEventPropLIF(unittest.TestCase):

    plot_path = Path(__file__).parent.joinpath("plots")

    def setUp(self):
        self.plot_path.mkdir(exist_ok=True)

    @staticmethod
    def _single_layer_network(backprop_method: str, v_reset=-1000.):
        params = LIFParameters(v_reset=v_reset, v_th=1.0)
        builder = Topology(
            t_max=50e-3,
            backprop_method=backprop_method,
        )
        builder.add(
            {
                "inp": Source(5),
                "lif": LIF(
                    100,
                    105,
                    params,
                ),
                "syn": Linear(
                    mean=3.0,
                    std=1.6,
                    min_delay=0.000,
                ),
            }
        )
        builder.connect(
            [
                ("inp", "syn"),
                ("syn", "lif"),
            ]
        )
        return builder.done()

    @staticmethod
    def _multi_layer_network(backprop_method: str, v_reset=-1000.):
        params = LIFParameters(v_reset=v_reset, v_th=1.0)
        builder = Topology(
            t_max=50e-3,
            backprop_method=backprop_method,
        )
        builder.add(
            {
                "inp": Source(5),
                "lif1": LIF(
                    100,
                    105,
                    params,
                ),
                "syn1": Linear(
                    mean=3.0,
                    std=1.6,
                    min_delay=0.000,
                ),
                "lif2": LIF(
                    5,
                    105,
                    params,
                ),
                "syn2": Linear(
                    mean=3.0,
                    std=1.6,
                    min_delay=0.000,
                ),
            }
        )
        builder.connect(
            [
                ("inp", "syn1"),
                ("syn1", "lif1"),
                ("lif1", "syn2"),
                ("syn2", "lif2"),
            ]
        )
        return builder.done()

    @staticmethod
    def _recurrent_network(backprop_method: str, v_reset=-1000.):
        params = LIFParameters(v_reset=v_reset, v_th=1.0)
        builder = Topology(
            t_max=50e-3,
            backprop_method=backprop_method,
        )
        builder.add(
            {
                "inp": Source(5),
                "lif": LIF(
                    100,
                    105,
                    params,
                ),
                "syn1": Linear(
                    mean=3.0,
                    std=1.6,
                    min_delay=0.000,
                ),
                "syn2": Linear(
                    mean=3.0,
                    std=1.6,
                    min_delay=0.000,
                ),
            }
        )
        builder.connect(
            [
                ("inp", "syn1"),
                ("syn1", "lif"),
                ("lif", "syn2"),
                ("syn2", "lif"),
            ]
        )
        return builder.done()

    @staticmethod
    def _loss_fn(
        apply_fn,
        key,
        weights,
        times,
        idxs,
        currents,
        layer_idx,
        internal,
    ):
        input_spikes = Spike(times, idxs, currents, layer_idx, internal)
        ret = apply_fn({"inp": input_spikes}, weights)
        return jnp.sum(jnp.where(ret[key].internal, ret[key].time, 0))

    def test_compare_single_layer_single_spike(self):
        """
        Note: Neurons only spike once
        """

        # declare net
        init_fn_1, apply_fn_1 = self._single_layer_network("analytical")
        # declare event prop net
        init_fn_2, apply_fn_2 = self._single_layer_network("eventprop")

        rng = jax.random.PRNGKey(42)
        params_1 = init_fn_1(rng)
        params_2 = init_fn_2(rng)

        # assert parameters equal
        self.assertIsNone(assert_array_equal(
            params_1["syn"],
            params_2["syn"],
        ))

        input_events = Spike(
            time=jnp.sort(np.random.uniform(0, 50e-3, (1, 10))),
            idx=np.random.uniform(0, 5, (1, 10)).astype(int),
            current=jnp.zeros((1, 10)),
            layer_idx=jnp.zeros((1, 10), dtype=int),
            internal=jnp.ones((1, 10), dtype=bool),
        )

        # forward
        res_1 = apply_fn_1({"inp": input_events}, params_1)["lif"]
        res_2 = apply_fn_2({"inp": input_events}, params_2)["lif"]

        # they all produce the same output
        self.assertIsNone(assert_array_equal(
            res_1.time,
            res_2.time,
        ))
        self.assertIsNone(assert_array_equal(
            res_1.idx,
            res_2.idx,
        ))
        self.assertIsNone(assert_array_equal(
            res_1.current,
            res_2.current,
        ))
        self.assertIsNone(assert_array_equal(
            res_1.layer_idx,
            res_2.layer_idx,
        ))
        self.assertIsNone(assert_array_equal(
            res_1.internal,
            res_2.internal,
        ))

        # now check grads
        loss_fn_1 = partial(self._loss_fn, apply_fn_1, "lif")
        loss_fn_2 = partial(self._loss_fn, apply_fn_2, "lif")

        loss_value_1 = loss_fn_1(
            params_1,
            input_events.time,
            input_events.idx,
            input_events.current,
            input_events.layer_idx,
            input_events.internal,
        )
        loss_value_2 = loss_fn_2(
            params_2,
            input_events.time,
            input_events.idx,
            input_events.current,
            input_events.layer_idx,
            input_events.internal,
        )

        self.assertIsNone(assert_array_equal(loss_value_1, loss_value_2))

        # check gradients
        _, (grad_w_1, grad_t_1)= jax.value_and_grad(
            loss_fn_1,
            argnums=(0, 1),
        )(
            params_1,
            input_events.time,
            input_events.idx,
            input_events.current,
            input_events.layer_idx,
            input_events.internal,
        )
        _, (grad_w_2, grad_t_2) = jax.value_and_grad(
            loss_fn_2,
            argnums=(0, 1),
        )(
            params_2,
            input_events.time,
            input_events.idx,
            input_events.current,
            input_events.layer_idx,
            input_events.internal,
        )

        _, axs = plt.subplots(nrows=4)
        axs[0].plot(grad_t_1.reshape(-1))
        axs[0].plot(grad_t_2.reshape(-1))
        axs[1].plot(jnp.abs(grad_t_1.reshape(-1) - grad_t_2.reshape(-1)))
        axs[0].set_ylabel("grad_t_1, grad_t_2")
        axs[1].set_ylabel("|grad_t_1 - grad_t_2|")
        axs[1].set_xlabel("Flat index")
        axs[2].plot(grad_w_1["syn"].reshape(-1))
        axs[2].plot(grad_w_2["syn"].reshape(-1))
        axs[3].plot(jnp.abs(grad_w_1["syn"] - grad_w_2["syn"]).reshape(-1))
        axs[2].set_ylabel("grad_w_1, grad_w_2")
        axs[3].set_ylabel("|grad_w_1 - grad_w_2|")
        axs[3].set_xlabel("Flat index")
        plt.savefig(self.plot_path.joinpath("./lif_vs_eventproplif_grads.png"))

        self.assertIsNone(assert_array_almost_equal(
            grad_t_1,
            grad_t_2,
            1,
        ))
        self.assertIsNone(assert_array_almost_equal(
            grad_w_1["syn"],
            grad_w_2["syn"],
            4,
        ))

        # check gradients when no spike
        zero_weights1 = tree_map(lambda p: p * 0.0, params_1)
        zero_weights2 = tree_map(lambda p: p * 0.0, params_2)

        # check gradients
        _, grad_w_1 = jax.value_and_grad(loss_fn_1)(
            zero_weights1,
            input_events.time,
            input_events.idx,
            input_events.current,
            input_events.layer_idx,
            input_events.internal,
        )
        _, grad_w_2 = jax.value_and_grad(loss_fn_2)(
            zero_weights2,
            input_events.time,
            input_events.idx,
            input_events.current,
            input_events.layer_idx,
            input_events.internal,
        )

        self.assertIsNone(assert_array_almost_equal(
            grad_w_1["syn"],
            grad_w_2["syn"],
            5,
        ))

    def test_compare_multiple_layers_single_spike(self):
        """
        Note: Neurons only spike once
        """
        # declare net
        init_fn_1, apply_fn_1 = self._multi_layer_network("analytical")
        # declare event prop net
        init_fn_2, apply_fn_2 = self._multi_layer_network("eventprop")

        rng = jax.random.PRNGKey(42)
        params_1 = init_fn_1(rng)
        params_2 = init_fn_2(rng)

        # assert parameters equal
        self.assertIsNone(assert_array_equal(
            params_1["syn1"],
            params_2["syn1"],
        ))
        # assert parameters equal
        self.assertIsNone(assert_array_equal(
            params_1["syn2"],
            params_2["syn2"],
        ))

        input_events = Spike(
            time=jnp.sort(np.random.uniform(0, 50e-3, (1, 10))),
            idx=np.random.uniform(0, 5, (1, 10)).astype(int),
            current=jnp.zeros((1, 10)),
            layer_idx=jnp.zeros((1, 10), dtype=int),
            internal=jnp.ones((1, 10), dtype=bool),
        )

        # forward
        res_1 = apply_fn_1({"inp": input_events}, params_1)
        res_2 = apply_fn_2({"inp": input_events}, params_2)

        res1_lif1 = res_1["lif1"]
        res1_lif2 = res_1["lif2"]
        res2_lif1 = res_2["lif1"]
        res2_lif2 = res_2["lif2"]

        # they all produce the same output
        self.assertIsNone(assert_array_equal(
            res1_lif1.time,
            res2_lif1.time,
        ))
        self.assertIsNone(assert_array_equal(
            res1_lif1.idx,
            res2_lif1.idx,
        ))
        self.assertIsNone(assert_array_equal(
            res1_lif1.current,
            res2_lif1.current,
        ))
        self.assertIsNone(assert_array_equal(
            res1_lif1.layer_idx,
            res2_lif1.layer_idx,
        ))
        self.assertIsNone(assert_array_equal(
            res1_lif1.internal,
            res2_lif1.internal,
        ))
        self.assertIsNone(assert_array_equal(
            res1_lif2.time,
            res2_lif2.time,
        ))
        self.assertIsNone(assert_array_equal(
            res1_lif2.idx,
            res2_lif2.idx,
        ))
        self.assertIsNone(assert_array_equal(
            res1_lif2.current,
            res2_lif2.current,
        ))
        self.assertIsNone(assert_array_equal(
            res1_lif2.layer_idx,
            res2_lif2.layer_idx,
        ))
        self.assertIsNone(assert_array_equal(
            res1_lif2.internal,
            res2_lif2.internal,
        ))

        # now check grads
        loss_fn_1 = partial(self._loss_fn, apply_fn_1, "lif2")
        loss_fn_2 = partial(self._loss_fn, apply_fn_2, "lif2")

        loss_value_1 = loss_fn_1(
            params_1,
            input_events.time,
            input_events.idx,
            input_events.current,
            input_events.layer_idx,
            input_events.internal,
        )
        loss_value_2 = loss_fn_2(
            params_2,
            input_events.time,
            input_events.idx,
            input_events.current,
            input_events.layer_idx,
            input_events.internal,
        )

        self.assertIsNone(assert_array_equal(loss_value_1, loss_value_2))

        # check gradients
        _, grad_1 = jax.value_and_grad(
            loss_fn_1,
        )(
            params_1,
            input_events.time,
            input_events.idx,
            input_events.current,
            input_events.layer_idx,
            input_events.internal,
        )
        _, grad_2  = jax.value_and_grad(
            loss_fn_2,
        )(
            params_2,
            input_events.time,
            input_events.idx,
            input_events.current,
            input_events.layer_idx,
            input_events.internal,
        )

        _, axs = plt.subplots(nrows=4)
        axs[0].plot(grad_1["syn1"].reshape(-1))
        axs[0].plot(grad_2["syn1"].reshape(-1))
        axs[1].plot(np.abs(grad_1["syn1"].reshape(-1) - grad_2["syn1"].reshape(-1)))
        axs[2].plot(grad_1["syn2"].reshape(-1))
        axs[2].plot(grad_2["syn2"].reshape(-1))
        axs[3].plot(np.abs(grad_1["syn2"].reshape(-1) - grad_2["syn2"].reshape(-1)))
        axs[0].set_ylabel("grad1, grad2")
        axs[1].set_ylabel("|grad1 - grad2|")
        axs[1].set_xlabel("Flat index")
        axs[2].set_ylabel("grad1, grad2")
        axs[3].set_ylabel("|grad1 - grad2|")
        axs[3].set_xlabel("Flat index")
        plt.savefig(self.plot_path.joinpath(
            "./multi_layer_lif_vs_eventproplif_grads.png"))

        self.assertIsNone(assert_array_almost_equal(
            grad_1["syn1"], grad_2["syn1"], 6),
        )
        self.assertIsNone(assert_array_almost_equal(
            grad_1["syn2"], grad_2["syn2"], 6),
        )

        # check gradients when no spike
        zero_weights1 = tree_map(lambda p: p * 0.0, params_1)
        zero_weights2 = tree_map(lambda p: p * 0.0, params_2)

        # check gradients
        _, grad_1 = jax.value_and_grad(loss_fn_1)(
            zero_weights1,
            input_events.time,
            input_events.idx,
            input_events.current,
            input_events.layer_idx,
            input_events.internal,
        )
        _, grad_2 = jax.value_and_grad(loss_fn_2)(
            zero_weights2,
            input_events.time,
            input_events.idx,
            input_events.current,
            input_events.layer_idx,
            input_events.internal,
        )

        self.assertIsNone(assert_array_almost_equal(
            grad_1["syn1"], grad_2["syn1"], 7)
        )
        self.assertIsNone(assert_array_almost_equal(
            grad_1["syn2"], grad_2["syn2"], 7)
        )

        # test multiple iterations
        for i in range(10):
            input_events = Spike(
                time=jnp.sort(np.random.uniform(0, 50e-3, (1, 10))),
                idx=np.random.uniform(0, 5, (1, 10)).astype(int),
                current=jnp.zeros((1, 10)),
                layer_idx=jnp.zeros((1, 10), dtype=int),
                internal=jnp.ones((1, 10), dtype=bool),
            )
            loss_value_1 = loss_fn_1(
                params_1,
                input_events.time,
                input_events.idx,
                input_events.current,
                input_events.layer_idx,
                input_events.internal,
            )
            loss_value_2 = loss_fn_2(
                params_2,
                input_events.time,
                input_events.idx,
                input_events.current,
                input_events.layer_idx,
                input_events.internal,
            )
            self.assertIsNone(assert_array_equal(loss_value_1, loss_value_2))

    def test_compare_single_layers_multiple_spike(self):
        """ """
        # declare net
        init_fn_1, apply_fn_1 = self._single_layer_network(
            "analytical",
            v_reset=0.0,
        )
        # decalre event prop net
        init_fn_2, apply_fn_2 = self._single_layer_network(
            "eventprop",
            v_reset=0.0,
        )

        rng = jax.random.PRNGKey(42)
        params_1 = init_fn_1(rng)
        params_2 = init_fn_2(rng)

        input_events = Spike(
            time=jnp.sort(np.random.uniform(0, 50e-3, (1, 10))),
            idx=np.random.uniform(0, 5, (1, 10)).astype(int),
            current=jnp.zeros((1, 10)),
            layer_idx=jnp.zeros((1, 10), dtype=int),
            internal=jnp.ones((1, 10), dtype=bool),
        )

        # forward
        res_1 = apply_fn_1({"inp": input_events}, params_1)["lif"]
        res_2 = apply_fn_2({"inp": input_events}, params_2)["lif"]

        # Make sure we actually have more then one spike for any neuron
        _, counts = jnp.unique(res_1.idx, return_counts=True)
        self.assertTrue(True in (counts > 1))

        # they all produce the same output
        self.assertIsNone(assert_array_equal(
            res_1.time,
            res_2.time,
        ))
        self.assertIsNone(assert_array_equal(
            res_1.idx,
            res_2.idx,
        ))
        self.assertIsNone(assert_array_equal(
            res_1.current,
            res_2.current,
        ))

        loss_fn_1 = partial(self._loss_fn, apply_fn_1, "lif")
        loss_fn_2 = partial(self._loss_fn, apply_fn_2, "lif")

        loss_value_1 = loss_fn_1(
            params_1,
            input_events.time,
            input_events.idx,
            input_events.current,
            input_events.layer_idx,
            input_events.internal,
        )
        loss_value_2 = loss_fn_2(
            params_2,
            input_events.time,
            input_events.idx,
            input_events.current,
            input_events.layer_idx,
            input_events.internal,
        )

        self.assertIsNone(assert_array_equal(loss_value_1, loss_value_2))

        # check gradients
        _, (grad_w_1, grad_t_1)= jax.value_and_grad(
            loss_fn_1,
            argnums=(0, 1),
        )(
            params_1,
            input_events.time,
            input_events.idx,
            input_events.current,
            input_events.layer_idx,
            input_events.internal,
        )
        _, (grad_w_2, grad_t_2) = jax.value_and_grad(
            loss_fn_2,
            argnums=(0, 1),
        )(
            params_2,
            input_events.time,
            input_events.idx,
            input_events.current,
            input_events.layer_idx,
            input_events.internal,
        )

        _, axs = plt.subplots(nrows=4)
        axs[0].plot(grad_t_1)
        axs[0].plot(grad_t_2)
        axs[1].plot(np.abs(grad_t_1 - grad_t_2))
        axs[0].set_ylabel("grad_t_1, grad_t_2")
        axs[1].set_ylabel("|grad_t_1 - grad_t_2|")
        axs[1].set_xlabel("Flat index")
        axs[2].plot(grad_w_1["syn"].reshape(-1))
        axs[2].plot(grad_w_2["syn"].reshape(-1))
        axs[3].plot(np.abs(grad_w_1["syn"] - grad_w_2["syn"]).reshape(-1))
        axs[2].set_ylabel("grad_w_1, grad_w_2")
        axs[3].set_ylabel("|grad_w_1 - grad_w_2|")
        axs[3].set_xlabel("Flat index")
        plt.savefig(self.plot_path.joinpath(
            "./lif_vs_eventproplif_grads_multiple_spikes.png"))

        self.assertIsNone(assert_array_almost_equal(
            grad_w_1["syn"],
            grad_w_2["syn"],
            5,
        ))
        self.assertIsNone(assert_array_almost_equal(
            grad_t_1,
            grad_t_2,
            1,
        ))
        self.assertIsNone(assert_array_almost_equal(
            grad_w_1["syn"],
            grad_w_2["syn"],
            4,
        ))

        # check gradients when no spike
        zero_weights1 = tree_map(lambda p: p * 0.0, params_1)
        zero_weights2 = tree_map(lambda p: p * 0.0, params_2)

        # check gradients
        _, grad_w_1 = jax.value_and_grad(loss_fn_1)(
            zero_weights1,
            input_events.time,
            input_events.idx,
            input_events.current,
            input_events.layer_idx,
            input_events.internal,
        )
        _, grad_w_2 = jax.value_and_grad(loss_fn_2)(
            zero_weights2,
            input_events.time,
            input_events.idx,
            input_events.current,
            input_events.layer_idx,
            input_events.internal,
        )

        self.assertIsNone(assert_array_almost_equal(
            grad_w_1["syn"],
            grad_w_2["syn"],
            5,
        ))

    def test_compare_multi_layers_multiple_spike(self):
        """ """
        # declare net
        init_fn_1, apply_fn_1 = self._multi_layer_network(
            "analytical",
            v_reset=0.0,
        )
        # decalre event prop net
        init_fn_2, apply_fn_2 = self._multi_layer_network(
            "eventprop",
            v_reset=0.0,
        )

        rng = jax.random.PRNGKey(42)
        params_1 = init_fn_1(rng)
        params_2 = init_fn_2(rng)

        # assert parameters equal
        self.assertIsNone(assert_array_equal(
            params_1["syn1"],
            params_2["syn1"],
        ))
        # assert parameters equal
        self.assertIsNone(assert_array_equal(
            params_1["syn2"],
            params_2["syn2"],
        ))

        input_events = Spike(
            time=jnp.sort(np.random.uniform(0, 50e-3, (1, 10))),
            idx=np.random.uniform(0, 5, (1, 10)).astype(int),
            current=jnp.zeros((1, 10)),
            layer_idx=jnp.zeros((1, 10), dtype=int),
            internal=jnp.ones((1, 10), dtype=bool),
        )

        # forward
        res_1 = apply_fn_1({"inp": input_events}, params_1)
        res_2 = apply_fn_2({"inp": input_events}, params_2)

        res1_lif1 = res_1["lif1"]
        res1_lif2 = res_1["lif2"]
        res2_lif1 = res_2["lif1"]
        res2_lif2 = res_2["lif2"]

        # Make sure we actually have more then one spike for any neuron
        _, counts = jnp.unique(res_1["lif1"].idx, return_counts=True)
        self.assertTrue(True in (counts > 1))

        # they all produce the same output
        self.assertIsNone(assert_array_equal(
            res1_lif1.time,
            res2_lif1.time,
        ))
        self.assertIsNone(assert_array_equal(
            res1_lif1.idx,
            res2_lif1.idx,
        ))
        self.assertIsNone(assert_array_equal(
            res1_lif1.current,
            res2_lif1.current,
        ))
        self.assertIsNone(assert_array_equal(
            res1_lif1.layer_idx,
            res2_lif1.layer_idx,
        ))
        self.assertIsNone(assert_array_equal(
            res1_lif1.internal,
            res2_lif1.internal,
        ))
        self.assertIsNone(assert_array_equal(
            res1_lif2.time,
            res2_lif2.time,
        ))
        self.assertIsNone(assert_array_equal(
            res1_lif2.idx,
            res2_lif2.idx,
        ))
        self.assertIsNone(assert_array_equal(
            res1_lif2.current,
            res2_lif2.current,
        ))
        self.assertIsNone(assert_array_equal(
            res1_lif2.layer_idx,
            res2_lif2.layer_idx,
        ))
        self.assertIsNone(assert_array_equal(
            res1_lif2.internal,
            res2_lif2.internal,
        ))

        # now check grads
        loss_fn_1 = partial(self._loss_fn, apply_fn_1, "lif2")
        loss_fn_2 = partial(self._loss_fn, apply_fn_2, "lif2")

        loss_value_1 = loss_fn_1(
            params_1,
            input_events.time,
            input_events.idx,
            input_events.current,
            input_events.layer_idx,
            input_events.internal,
        )
        loss_value_2 = loss_fn_2(
            params_2,
            input_events.time,
            input_events.idx,
            input_events.current,
            input_events.layer_idx,
            input_events.internal,
        )

        self.assertIsNone(assert_array_equal(loss_value_1, loss_value_2))

        # check gradients
        _, grad_1 = jax.value_and_grad(
            loss_fn_1,
        )(
            params_1,
            input_events.time,
            input_events.idx,
            input_events.current,
            input_events.layer_idx,
            input_events.internal,
        )
        _, grad_2  = jax.value_and_grad(
            loss_fn_2,
        )(
            params_2,
            input_events.time,
            input_events.idx,
            input_events.current,
            input_events.layer_idx,
            input_events.internal,
        )

        _, axs = plt.subplots(nrows=4)
        axs[0].plot(grad_1["syn1"].reshape(-1))
        axs[0].plot(grad_2["syn1"].reshape(-1))
        axs[1].plot(np.abs(grad_1["syn1"] - grad_2["syn1"]).reshape(-1))
        axs[2].plot(grad_1["syn2"].reshape(-1))
        axs[2].plot(grad_2["syn2"].reshape(-1))
        axs[3].plot(np.abs(grad_1["syn2"] - grad_2["syn2"]).reshape(-1))
        axs[0].set_ylabel("grad1, grad2")
        axs[1].set_ylabel("|grad1 - grad2|")
        axs[1].set_xlabel("Flat index")
        axs[2].set_ylabel("grad1, grad2")
        axs[3].set_ylabel("|grad1 - grad2|")
        axs[3].set_xlabel("Flat index")
        plt.savefig(self.plot_path.joinpath(
            "./multi_layer_lif_vs_eventproplif_grads_multiple_spikes.png"))

        self.assertIsNone(assert_array_almost_equal(
            grad_1["syn1"], grad_2["syn1"], 6),
        )
        self.assertIsNone(assert_array_almost_equal(
            grad_1["syn2"], grad_2["syn2"], 6),
        )

        # check gradients when no spike
        zero_weights1 = tree_map(lambda p: p * 0.0, params_1)
        zero_weights2 = tree_map(lambda p: p * 0.0, params_2)

        # check gradients
        _, grad_1 = jax.value_and_grad(loss_fn_1)(
            zero_weights1,
            input_events.time,
            input_events.idx,
            input_events.current,
            input_events.layer_idx,
            input_events.internal,
        )
        _, grad_2 = jax.value_and_grad(loss_fn_2)(
            zero_weights2,
            input_events.time,
            input_events.idx,
            input_events.current,
            input_events.layer_idx,
            input_events.internal,
        )

        self.assertIsNone(assert_array_almost_equal(
            grad_1["syn1"], grad_2["syn1"], 7)
        )
        self.assertIsNone(assert_array_almost_equal(
            grad_1["syn2"], grad_2["syn2"], 7)
        )

        # test multiple iterations
        for i in range(10):
            input_events = Spike(
                time=jnp.sort(np.random.uniform(0, 50e-3, (1, 10))),
                idx=np.random.uniform(0, 5, (1, 10)).astype(int),
                current=jnp.zeros((1, 10)),
                layer_idx=jnp.zeros((1, 10), dtype=int),
                internal=jnp.ones((1, 10), dtype=bool),
            )
            loss_value_1 = loss_fn_1(
                params_1,
                input_events.time,
                input_events.idx,
                input_events.current,
                input_events.layer_idx,
                input_events.internal,
            )
            loss_value_2 = loss_fn_2(
                params_2,
                input_events.time,
                input_events.idx,
                input_events.current,
                input_events.layer_idx,
                input_events.internal,
            )
            self.assertIsNone(assert_array_equal(loss_value_1, loss_value_2))

    def test_recurrent_multiple_spike(self):
        """ Note: Neurons spike several times """
        # declare net
        init_fn_1, apply_fn_1 = self._recurrent_network(
            "analytical",
            v_reset=0.0,
        )
        # decalre event prop net
        init_fn_2, apply_fn_2 = self._recurrent_network(
            "eventprop",
            v_reset=0.0,
        )

        rng = jax.random.PRNGKey(42)
        params_1 = init_fn_1(rng)
        params_2 = init_fn_2(rng)

        # assert parameters equal
        self.assertIsNone(assert_array_equal(
            params_1["syn1"],
            params_2["syn1"],
        ))
        # assert parameters equal
        self.assertIsNone(assert_array_equal(
            params_1["syn2"],
            params_2["syn2"],
        ))

        input_events = Spike(
            time=jnp.sort(np.random.uniform(0, 50e-3, (1, 10))),
            idx=np.random.uniform(0, 5, (1, 10)).astype(int),
            current=jnp.zeros((1, 10)),
            layer_idx=jnp.zeros((1, 10), dtype=int),
            internal=jnp.ones((1, 10), dtype=bool),
        )

        # forward
        res_1 = apply_fn_1({"inp": input_events}, params_1)["lif"]
        res_2 = apply_fn_2({"inp": input_events}, params_2)["lif"]

        # Make sure we actually have more then one spike for any neuron
        _, counts = jnp.unique(res_1.idx, return_counts=True)
        self.assertTrue(True in (counts > 1))

        # they all produce the same output
        self.assertIsNone(assert_array_equal(
            res_1.time,
            res_2.time,
        ))
        self.assertIsNone(assert_array_equal(
            res_1.idx,
            res_2.idx,
        ))
        self.assertIsNone(assert_array_equal(
            res_1.current,
            res_2.current,
        ))
        self.assertIsNone(assert_array_equal(
            res_1.layer_idx,
            res_2.layer_idx,
        ))
        self.assertIsNone(assert_array_equal(
            res_1.internal,
            res_2.internal,
        ))

        # now check grads
        loss_fn_1 = partial(self._loss_fn, apply_fn_1, "lif")
        loss_fn_2 = partial(self._loss_fn, apply_fn_2, "lif")

        loss_value_1 = loss_fn_1(
            params_1,
            input_events.time,
            input_events.idx,
            input_events.current,
            input_events.layer_idx,
            input_events.internal,
        )
        loss_value_2 = loss_fn_2(
            params_2,
            input_events.time,
            input_events.idx,
            input_events.current,
            input_events.layer_idx,
            input_events.internal,
        )

        self.assertIsNone(assert_array_equal(loss_value_1, loss_value_2))

        # check gradients
        _, (grad_w_1, grad_t_1)= jax.value_and_grad(
            loss_fn_1,
            argnums=(0, 1),
        )(
            params_1,
            input_events.time,
            input_events.idx,
            input_events.current,
            input_events.layer_idx,
            input_events.internal,
        )
        _, (grad_w_2, grad_t_2) = jax.value_and_grad(
            loss_fn_2,
            argnums=(0, 1),
        )(
            params_2,
            input_events.time,
            input_events.idx,
            input_events.current,
            input_events.layer_idx,
            input_events.internal,
        )

        _, axs = plt.subplots(nrows=4)
        axs[0].plot(grad_w_1["syn1"].reshape(-1))
        axs[0].plot(grad_w_2["syn1"].reshape(-1))
        axs[1].plot(np.abs(grad_w_1["syn1"] - grad_w_2["syn1"]).reshape(-1))
        axs[2].plot(grad_w_1["syn2"].reshape(-1))
        axs[2].plot(grad_w_2["syn2"].reshape(-1))
        axs[3].plot(np.abs(grad_w_1["syn2"] - grad_w_2["syn2"]).reshape(-1))
        axs[0].set_ylabel("grad1, grad2")
        axs[1].set_ylabel("|grad1 - grad2|")
        axs[1].set_xlabel("Flat index")
        axs[2].set_ylabel("grad1, grad2")
        axs[3].set_ylabel("|grad1 - grad2|")
        axs[3].set_xlabel("Flat index")
        plt.savefig(
            self.plot_path.joinpath("./lif_vs_entproplif_grads_recurrent.png"))

        self.assertIsNone(assert_array_almost_equal(
            grad_w_1["syn1"],
            grad_w_2["syn1"],
            4,
        ))

        # check gradients when no spike
        zero_weights1 = tree_map(lambda p: p * 0.0, params_1)
        zero_weights2 = tree_map(lambda p: p * 0.0, params_2)

        # check gradients
        _, grad_1 = jax.value_and_grad(loss_fn_1)(
            zero_weights1,
            input_events.time,
            input_events.idx,
            input_events.current,
            input_events.layer_idx,
            input_events.internal,
        )
        _, grad_2 = jax.value_and_grad(loss_fn_2)(
            zero_weights2,
            input_events.time,
            input_events.idx,
            input_events.current,
            input_events.layer_idx,
            input_events.internal,
        )

        self.assertIsNone(assert_array_almost_equal(
            grad_1["syn1"], grad_2["syn1"], 7)
        )
        self.assertIsNone(assert_array_almost_equal(
            grad_1["syn2"], grad_2["syn2"], 7)
        )


if __name__ == "__main__":
    unittest.main()
