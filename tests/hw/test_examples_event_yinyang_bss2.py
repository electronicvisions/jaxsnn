import unittest
import os

from functools import partial
import matplotlib.pyplot as plt
import numpy as np

import jax
from jax import random
import jax.numpy as jnp

import jaxsnn
from jaxsnn.examples.event import yinyang_bss2
from jaxsnn.event.types import Spike
from jaxsnn.base.dataset import yinyang_dataset
from jaxsnn.event.encode import (
    spatio_temporal_encode,
    target_temporal_encode,
    encode,
)
from jaxsnn.event.hardware.parameter import (
    MixedHXModelParameter,
    HXParameter,
)
from jaxsnn.event.modules.hx import (
    HXSource,
    HXLIF,
    NeuronParameters,
    HXLinear,
)
from jaxsnn.event.topology import Topology


class TestYinYangExample(unittest.TestCase):
    """ Tests the YinYang example implementation """

    def setUp(self):
        jaxsnn.init_hardware()

    def tearDown(self):
        jaxsnn.release_hardware()

    def test_training(self) -> None:
        """ Run YinYang training and inference """

        # run training
        train_args = [
            "--seed=0",
            "--duplicate-neurons",
            "--testset-size=2944",
            "--trainset-size=4992",
            "--t-late=1.2e-05",
            "--correct-target-time=5.4e-06",
            "--wrong-target-time=6.6e-06",
            "--tau-mem=1.2e-05",
            "--tau-syn=6e-06",
            "--n-spikes-hidden=100",
            "--epochs=10",
            "--batch-size=64",
            "--lr=0.005",
            "--lr-decay=0.98",
            "--duplication=5",
            "--hidden-size=100",
            "--hw-correction=0",
            "--max-runtime=30e-6",
            "--weight-scale=58",
        ]

        accuracy = yinyang_bss2.main(
            yinyang_bss2.get_parser().parse_args(train_args)
        )
        self.assertGreater(
            accuracy,
            0.85,
            "Accuracy should be greater than 85%",
        )


class YinYangObsvTest(unittest.TestCase):
    """ Tests the YinYang example implementation """

    tau_mem = 12e-6
    tau_syn = 6e-6
    runtime = 50e-6
    input_size = 5
    hidden_size = 100
    output_size = 3
    duplication = 5
    weight_scale = 56

    def test_topology(self):
        jaxsnn.init_hardware()

        # neuron params, low v_reset only allows one spike per neuron
        lif_params = NeuronParameters(
            v_th=MixedHXModelParameter(0.6, 150),
            v_leak=MixedHXModelParameter(0.0, 80),
            v_reset=MixedHXModelParameter(0.0, 80),
            refractory_time=HXParameter(30e-6),
            i_synin_gm=HXParameter(500),
            synapse_dac_bias=HXParameter(1000),
            tau_syn=HXParameter(self.tau_syn),
            tau_mem=HXParameter(self.tau_mem),
        )

        # define trainset and testset
        rng = random.PRNGKey(42)
        param_rng, rng = random.split(rng, 2)

        # define topology
        builder = Topology(
            mock=False,
            t_max=50e-6,
            backprop_method="eventprop",
        )

        # create modules
        builder.add(
            {
                "inp": HXSource(
                    size=25,
                    n_events=5,
                ),
                "lif_h": HXLIF(
                    size=self.hidden_size,
                    n_steps=self.hidden_size + 5 * self.duplication,
                    n_hw_spikes=self.hidden_size,
                    params=lif_params,
                    time_offset=-20,
                ),
                "lif_o": HXLIF(
                    size=3,
                    n_steps=3 + self.hidden_size,
                    n_hw_spikes=self.hidden_size,
                    params=lif_params,
                    time_offset=-20,
                ),
                "syn_ih": HXLinear(
                    mean=0.6,
                    std=0.3,
                    min_delay=0.0,
                    weight_scale=self.weight_scale,
                ),
                "syn_ho": HXLinear(
                    mean=0.5,
                    std=0.8,
                    min_delay=0.0,
                    weight_scale=self.weight_scale,
                )
            }
        )

        # connect modules
        builder.connect(
            [
                ("inp", "syn_ih"),
                ("syn_ih", "lif_h"),
                ("lif_h", "syn_ho"),
                ("syn_ho", "lif_o"),
            ]
        )

        inputs = {
            "inp": Spike(
                time=3e-6 * jnp.ones((1, 5)),
                idx=jnp.arange(5, dtype=jnp.int32).reshape(1, -1),
                internal=jnp.ones((1, 5), dtype=bool),
                layer_idx=jnp.zeros((1, 5), dtype=jnp.int32),
                current=jnp.zeros((1, 5)),
        )}

        # build topology
        init_fn, apply_fn = builder.done()
        params = init_fn(param_rng)

        # Test spikes on hidden layer
        params["syn_ho"] = params["syn_ho"].at[:, :].set(0)
        for i in range(self.hidden_size):
            params["syn_ih"] = params["syn_ih"].at[:, :].set(0)
            params["syn_ih"] = params["syn_ih"].at[:, i].set(64)
            # Forward
            spikes = apply_fn(
                inputs,
                params,
            )["lif_h"]
            spikes_idx = spikes.idx[spikes.internal]
            idxs = jnp.unique(spikes_idx)
            self.assertEqual(len(idxs), 1)
            self.assertEqual(idxs[0], i)

        # Test spikes on ouput layer
        params["syn_ho"] = params["syn_ho"].at[:, :].set(0)
        params["syn_ih"] = params["syn_ih"].at[:, :].set(0)
        params["syn_ih"] = params["syn_ih"].at[:, :5].set(64)
        for i in range(self.output_size):
            params["syn_ho"] = params["syn_ho"].at[:, :].set(0)
            params["syn_ho"] = params["syn_ho"].at[:5, i].set(64)
            # Forward
            spikes = apply_fn(
                inputs,
                params,
            )["lif_o"]
            spikes_idx = spikes.idx[spikes.internal]
            idxs = jnp.unique(spikes_idx)
            self.assertEqual(len(idxs), 1)
            self.assertEqual(idxs[0], i)

    def test_validate_hw_observables(self):
        jaxsnn.init_hardware()
        # neuron params, low v_reset only allows one spike per neuron
        lif_params = NeuronParameters(
            v_th=MixedHXModelParameter(0.6, 150),
            v_leak=MixedHXModelParameter(0.0, 80),
            v_reset=MixedHXModelParameter(0.0, 80),
            refractory_time=HXParameter(30e-6),
            i_synin_gm=HXParameter(500),
            synapse_dac_bias=HXParameter(1000),
            tau_syn=HXParameter(self.tau_syn),
            tau_mem=HXParameter(self.tau_mem),
        )

        t_max = 4.0 * lif_params.tau_syn.model_value
        t_max = max(t_max, self.runtime)

        # How many input neurons do we have?
        input_size = self.input_size * self.duplication

        # define trainset and testset
        rng = random.PRNGKey(42)
        param_rng, data_rng, rng = random.split(rng, 3)

        xy_dataset = yinyang_dataset(
            data_rng, 10, mirror=True, bias_spike=0.0,
        )

        # Encoding
        input_encoder_batched = jax.jit(jax.vmap(partial(
            spatio_temporal_encode,
            t_late=2 * 6e-6,
            duplication=5,
            duplicate_neurons=True,
        )))
        target_encoder_batched = jax.jit(jax.vmap(partial(
            target_temporal_encode,
            n_classes=3,
            correct_target_time=0.9 * 6e-6,
            wrong_target_time=1.1 * 6e-6,
        )))

        # Datasets
        dataset = encode(
            xy_dataset,
            input_encoder_batched,
            target_encoder_batched,
        )

        dataset = ({"inp": dataset[0]}, dataset[1])
        # dataset_batched = data_loader(dataset, 10)

        # define topology
        builder = Topology(
            mock=False,
            t_max=t_max,
            backprop_method="eventprop",
        )

        # create modules
        builder.add(
            {
                "inp": HXSource(
                    size=input_size,
                    n_events=input_size,
                ),
                "lif_h": HXLIF(
                    size=self.hidden_size,
                    n_steps=self.hidden_size + 5 * self.duplication,
                    n_hw_spikes=self.hidden_size,
                    params=lif_params,
                    time_offset=0,
                ),
                "lif_o": HXLIF(
                    size=3,
                    n_steps=3 + self.hidden_size,
                    n_hw_spikes=self.hidden_size,
                    params=lif_params,
                    time_offset=0,
                ),
                "syn_ih": HXLinear(
                    mean=0.6,
                    std=0.3,
                    min_delay=0.0,
                    weight_scale=self.weight_scale,
                ),
                "syn_ho": HXLinear(
                    mean=0.5,
                    std=0.8,
                    min_delay=0.0,
                    weight_scale=self.weight_scale,
                )
            }
        )

        # connect modules
        builder.connect(
            [
                ("inp", "syn_ih"),
                ("syn_ih", "lif_h"),
                ("lif_h", "syn_ho"),
                ("syn_ho", "lif_o"),
            ]
        )

        # build topology
        init_fn, apply_fn = builder.done()

        # init weights
        params = init_fn(param_rng)

        # Forward
        spikes = apply_fn(
            dataset[0],
            params,
        )

        # Call the merged plotting function
        plot_network_activity(
            dataset[0]["inp"],
            dataset[1],
            spikes,
        )

        jaxsnn.release_hardware()


def plot_network_activity(
    input_spikes,
    targets,
    network_spikes,
    batch_entry=0,
):
    """
    Plots input spikes, target times, and layer spikes in a single figure.
    Only plots internal spikes for hidden and output layers.
    """
    # Input and Targets
    input_times = np.asarray(input_spikes.time[batch_entry])
    input_indices = np.asarray(input_spikes.idx[batch_entry])
    target_times = np.asarray(targets[batch_entry])

    # Hidden Layer
    hidden_spikes = network_spikes['lif_h'].get_internal().sort(-1)
    hidden_times = np.asarray(hidden_spikes.time[batch_entry])
    hidden_indices = np.asarray(hidden_spikes.idx[batch_entry])
    valid_hidden_mask = hidden_spikes.internal[batch_entry]

    # Output Layer
    output_spikes = network_spikes['lif_o'].get_internal().sort(-1)
    output_times = np.asarray(output_spikes.time[batch_entry])
    output_indices = np.asarray(output_spikes.idx[batch_entry])
    valid_output_mask = output_spikes.internal[batch_entry]

    # Create Plot
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(15, 15), sharex=True,
    )
    fig.suptitle(
        f'Network Activity for Batch Entry {batch_entry}', fontsize=16,
    )

    # 1. Input Spikes and Targets
    ax1.scatter(
        input_times,
        input_indices,
        marker='|',
        color='black',
        label='Input Spikes',
    )
    colors = ['r', 'g', 'b']
    for i, target_time in enumerate(target_times):
        ax1.axvline(
            x=target_time,
            color=colors[i],
            linestyle='--',
            label=f'Target Class {i}',
        )
    ax1.set_title('Input Layer and Target Times')
    ax1.set_ylabel('Neuron Index')
    ax1.legend()
    ax1.grid(True, axis='x', linestyle=':', alpha=0.6)

    # 2. Hidden Layer Spikes
    ax2.scatter(
        hidden_times[valid_hidden_mask],
        hidden_indices[valid_hidden_mask],
        marker='|',
        color='blue'
    )
    ax2.set_title('Hidden Layer Internal Spikes')
    ax2.set_ylabel('Neuron Index')
    ax2.grid(True, axis='x', linestyle=':', alpha=0.6)

    # 3. Output Layer Spikes
    ax3.scatter(
        output_times[valid_output_mask],
        output_indices[valid_output_mask],
        marker='|',
        color='green'
    )
    ax3.set_title('Output Layer Internal Spikes')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Neuron Index')
    ax3.grid(True, axis='x', linestyle=':', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plots_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_filename = os.path.join(
        plots_dir,
        f"network_activity_batch_entry.png",
    )
    plt.savefig(plot_filename)
    plt.close()


if __name__ == "__main__":
    unittest.main()
