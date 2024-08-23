# pylint: disable=invalid-name,line-too-long,too-many-arguments,too-many-locals,unsubscriptable-object,unused-argument,undefined-variable
import logging
from typing import List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from jaxsnn.base.types import Array
from jaxsnn.event.dataset.utils import Dataset
from jaxsnn.event.hardware.utils import filter_spikes
from jaxsnn.event.types import Spike, TestResult, Weight, WeightRecurrent

log = logging.getLogger(__name__)

blue = np.array([[47, 66, 87, 210]]) / 256
red = np.array([[103, 43, 40, 210]]) / 256
black = np.array([[0, 0, 0, 0.3]])
green = np.array([[47, 75, 37, 210]]) / 256


def plt_spikes(
    ax: List[plt.Axes],
    recording,
    sizes: Tuple[int],
    mock: bool,
    tau_syn: float,
):
    # plt five different samples
    for j, plt_idx in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        spikes = recording[-1][-1, 0, j]
        input_size, hidden_size, _ = sizes
        layers = [
            filter_spikes(spikes, 0, input_size),
            filter_spikes(spikes, input_size, hidden_size + input_size),
            filter_spikes(spikes, hidden_size + input_size),
        ]

        unit = tau_syn if mock else 1e-6

        for i in range(3):
            ax[plt_idx].scatter(
                x=layers[i].time / unit,
                y=layers[i].idx,
                s=(20, 10, 20)[i],
                c=("green", "black", "orange")[i],
                label=f"Layer {i}",
            )
        ax[plt_idx].set_ylabel("neuron id")
        if mock:
            ax[plt_idx].set_xlabel(r"$t [\tau_s]$")
        else:
            ax[plt_idx].set_xlabel(r"$t$ [us]")
        ax[plt_idx].legend()
        ax[plt_idx].set_title(f"Spike times in {'SW' if mock else 'HW'}")


def plt_loss(ax: plt.Axes, loss: Array):
    ax.plot(np.arange(len(loss)), loss)
    ax.set_ylabel("test loss")


def plt_accuracy(ax: plt.Axes, accuracy: Array):
    ax.plot(np.arange(len(accuracy)), accuracy)
    ax.set_ylabel("test accuracy")


def plt_no_spike_prob(ax: List[plt.Axes], t_spike, testset):
    output_size = testset[1].shape[-1]
    for neuron_idx in range(output_size):
        for which_class in range(output_size):
            samples = t_spike[
                :, np.argmin(testset[1], axis=-1) == which_class, neuron_idx
            ]
            prob_no_spike = np.mean(samples == np.inf, axis=1)
            ax[neuron_idx].plot(
                np.arange(len(prob_no_spike)),
                prob_no_spike,
                label=f"Class {which_class + 1}",
            )

        ax[neuron_idx].title.set_text(f"Neuron {neuron_idx + 1}")
        ax[neuron_idx].set_ylabel("prob no spike")
        ax[neuron_idx].legend()


def plt_average_spike_time(
    ax: plt.Axes,
    t_spike_correct: Array,
    t_spike_false: Array,
    const_target: Array,
):
    t_spike_correct_avg = np.nanmean(t_spike_correct, axis=(1, 2))
    t_spike_false_avg = np.nanmean(t_spike_false, axis=(1, 2))

    ax.plot(
        np.arange(len(t_spike_correct_avg)),
        t_spike_correct_avg,
        label="Correct neuron",
    )
    ax.axhline(const_target[0], color="red")

    # plot spike time non-correct neuron
    ax.plot(
        np.arange(len(t_spike_false_avg)),
        t_spike_false_avg,
        label="False neuron",
    )
    ax.axhline(const_target[2], color="red")

    ax.set_xlabel("Epoch")
    ax.title.set_text("Output spike times")
    ax.legend()


def plt_circle(ax: plt.Axes, radius: float, offset: float):
    theta = np.linspace(0, 2 * np.pi, 150)
    a = radius * np.cos(theta)
    b = radius * np.sin(theta)
    ax.plot(a + offset, b + offset, c="k")


def plt_prediction(
    ax: plt.Axes,
    dataset_in: Dataset,
    t_spike: Array,
    tau_syn: float,
    duplication: int = 1,
    duplicate_neurons: bool = False,
):
    t_late = 2.0

    # reshape
    n_classes = dataset_in[1].shape[-1]
    input_size = dataset_in[0].time.shape[-1]
    dataset = (
        Spike(
            dataset_in[0].time.reshape(-1, input_size)[:, ::duplication],
            dataset_in[0].idx.reshape(-1, input_size)[:, ::duplication],
        ),
        dataset_in[1].reshape(-1, n_classes),
        dataset_in[2],
    )

    x_ix = np.argwhere(dataset[0].idx == 0)
    x = dataset[0].time[x_ix[:, 0], x_ix[:, 1]] / tau_syn

    if duplicate_neurons:
        y_ix = np.argwhere(dataset[0].idx == duplication)
    else:
        y_ix = np.argwhere(dataset[0].idx == 1)

    y = dataset[0].time[y_ix[:, 0], y_ix[:, 1]] / tau_syn

    pred_class = np.argmin(t_spike[-1], axis=-1).flatten()
    for i, color in zip(range(n_classes), (blue, red, green)):
        ax.scatter(
            x[pred_class == i],
            y[pred_class == i],
            s=30,
            facecolor=color,
            edgecolor=black,
            linewidths=0.7,
        )
    if dataset[2] == "circle":
        radius = np.sqrt(0.5 / np.pi) * t_late
        plt_circle(ax, radius=radius, offset=0.75)
    elif dataset[2] == "linear":
        ax.plot(
            np.linspace(0, t_late, 100), np.linspace(0, t_late, 100), c="k"
        )
    ax.set_xlabel(r"Input time $t_x [\tau_s]$")
    ax.set_ylabel(r"Input time $t_y [\tau_s]$")
    ax.set_xticks(np.linspace(0.0, t_late, 6))
    ax.set_yticks(np.linspace(0.0, t_late, 6))
    ax.title.set_text("Classification")
    ax.set_aspect("equal")


def plt_t_spike_neuron(
    fig,
    axs: List[plt.Axes],
    dataset_in: Dataset,
    t_spike: Array,
    tau_syn: float,
    duplication: int = 1,
    duplicate_neurons: bool = False,
):
    t_late = 2.0
    names = ("First", "Second", "Third")
    n_neurons = t_spike[-1].shape[-1]
    t_spike = t_spike[-1].reshape(-1, n_neurons)

    # reshape
    n_classes = dataset_in[1].shape[-1]
    input_size = dataset_in[0].time.shape[-1]
    dataset = (
        Spike(
            dataset_in[0].time.reshape(-1, input_size)[:, ::duplication],
            dataset_in[0].idx.reshape(-1, input_size)[:, ::duplication],
        ),
        dataset_in[1].reshape(-1, n_classes),
        dataset_in[2],
    )

    x_ix = np.argwhere(dataset[0].idx == 0)
    x = dataset[0].time[x_ix[:, 0], x_ix[:, 1]] / tau_syn

    if duplicate_neurons:
        y_ix = np.argwhere(dataset[0].idx == duplication)
    else:
        y_ix = np.argwhere(dataset[0].idx == 1)

    y = dataset[0].time[y_ix[:, 0], y_ix[:, 1]] / tau_syn

    # normalize all neurons with the same value
    cmap = mpl.colormaps["magma"]
    normalize = np.nanpercentile(
        np.where(t_spike == np.inf, np.nan, t_spike), 95
    )
    for i in range(n_neurons):
        score = t_spike[:, i]
        score_no_inf = score[score != np.inf]
        color_ix = np.rint(score_no_inf * 256 / normalize).astype(int)
        colors = cmap(color_ix)

        axs[i].scatter(
            x[score != np.inf],
            y[score != np.inf],
            s=30,
            facecolor=colors,
            edgecolor=black,
            linewidths=0.7,
        )
        axs[i].scatter(
            x[score == np.inf],
            y[score == np.inf],
            facecolor=black,
            edgecolor=black,
        )
        if dataset[2] == "linear":
            axs[i].plot(
                np.linspace(0, t_late, 100), np.linspace(0, t_late, 100), c="k"
            )
        elif dataset[2] == "circle":
            radius = np.sqrt(0.5 / np.pi) * t_late
            plt_circle(axs[i], radius=radius, offset=0.75)
        axs[i].set_xlabel(r"Input time $t_x [\tau_s]$")
        axs[i].set_xticks(np.linspace(0, t_late, 5))
        axs[i].set_yticks(np.linspace(0, t_late, 5))
        if i == 0:
            axs[i].set_ylabel(r"Input time $t_y [\tau_s]$")
        axs[i].title.set_text(f"{names[i]} neuron")
        axs[i].set_aspect("equal")

    # Normalizer
    norm = mpl.colors.Normalize(vmin=0, vmax=normalize / tau_syn)

    # creating ScalarMappable
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=axs, location="right")
    cbar.ax.set_ylabel(r"$t_{spike}[\tau_s]$", rotation=90, fontsize=13)
    fig.subplots_adjust(bottom=0.15, right=0.7, top=0.9)


def plt_dataset(
    ax: plt.Axes,
    dataset_in: Dataset,
    tau_syn: float,
    observe: Optional[Tuple[Tuple[int, int, str], ...]] = None,
    duplication: int = 1,
    duplicate_neurons: bool = False,
):
    t_late = 2.0
    n_classes = dataset_in[1].shape[-1]
    input_size = dataset_in[0].time.shape[-1]
    dataset = (
        Spike(
            dataset_in[0].time.reshape(-1, input_size)[:, ::duplication],
            dataset_in[0].idx.reshape(-1, input_size)[:, ::duplication],
        ),
        dataset_in[1].reshape(-1, n_classes),
        dataset_in[2],
    )

    x_ix = np.argwhere(dataset[0].idx == 0)
    x = dataset[0].time[x_ix[:, 0], x_ix[:, 1]] / tau_syn

    if duplicate_neurons:
        y_ix = np.argwhere(dataset[0].idx == duplication)
    else:
        y_ix = np.argwhere(dataset[0].idx == 1)

    y = dataset[0].time[y_ix[:, 0], y_ix[:, 1]] / tau_syn

    label = np.argmin(dataset[1], axis=-1).flatten()
    for i, color in zip(range(n_classes), (blue, red, green)):
        ax.scatter(
            x[label == i],
            y[label == i],
            s=30,
            facecolor=color,
            edgecolor=black,
            linewidths=0.7,
        )
    if observe is not None:
        for ix1, ix2, marker in observe:
            # TODO this calculation is wrong
            ix = ix1 * ix2
            ax.scatter(x[ix], y[ix], marker=marker, color="yellow", s=50)
    if dataset[2] == "circle":
        radius = np.sqrt(0.5 / np.pi) * t_late
        plt_circle(ax, radius=radius, offset=0.75)
    elif dataset[2] == "linear":
        ax.plot(
            np.linspace(0, t_late, 100), np.linspace(0, t_late, 100), c="k"
        )
    ax.set_xlabel(r"Input time $t_x [\tau_s]$")
    ax.set_ylabel(r"Input time $t_y [\tau_s]$")
    ax.set_xticks(np.linspace(0.0, t_late, 6))
    ax.set_yticks(np.linspace(0.0, t_late, 6))
    ax.title.set_text("Dataset")
    ax.set_aspect("equal")


def plt_2dloss(
    axs: plt.Axes,
    t_spike: Array,
    dataset: Array,
    observe: Tuple[Tuple[int, int, str], ...],
    tau_syn: float,
):
    for i, (ix1, ix2, marker) in enumerate(observe):
        trajectory = t_spike[:, ix1, ix2, :]
        target = dataset[1][ix1, ix2]
        n = 100

        def loss_fn_vec(t1, t2, target_1, target_2, tau_syn):
            idx = np.argmin(np.array([target_1, target_2]))
            first_spikes = np.array([t1, t2])
            alpha = 1.0
            zaehler = alpha + np.exp(-first_spikes[idx] / tau_syn)
            nenner = 1 + np.exp(-np.abs(first_spikes) / tau_syn)

            loss_value = -np.log(zaehler / np.sum(nenner))
            return loss_value

        # units of tau syn
        t1 = np.linspace(0, 2, n)
        t2 = np.linspace(0, 2, n)
        xx, yy = np.meshgrid(t1, t2)
        zz = loss_fn_vec(
            xx * tau_syn, yy * tau_syn, target[0], target[1], tau_syn
        )

        clipped = np.clip(trajectory, 0, 2 * tau_syn)
        axs[0, i].contourf(
            t1, t2, zz, levels=500, cmap=mpl.colormaps["viridis"]
        )
        axs[0, i].scatter(
            clipped[:, 0] / tau_syn,
            clipped[:, 1] / tau_syn,
            s=np.linspace(2, 8, clipped.shape[0]),
            color="green",
        )
        axs[0, i].axis("scaled")
        axs[0, i].set_xlabel(r"spike time neuron 1 $[\tau_s]$")
        if i == 0:
            axs[0, i].set_ylabel(r"spike time neuron 2 $[\tau_s]$")
        axs[0, i].set_xticks(np.linspace(0, 2, 5))
        axs[0, i].set_yticks(np.linspace(0, 2, 5))

        # plot marker
        axs[0, i].scatter(0.3, 1.8, marker=marker, color="yellow", s=50)
        axs[1, i].scatter(5, 1.8, marker=marker, color="green", s=50)

        axs[1, i].plot(
            np.arange(trajectory.shape[0]),
            trajectory[:, 0] / tau_syn,
            label="Neuron 1",
        )
        axs[1, i].plot(
            np.arange(trajectory.shape[0]),
            trajectory[:, 1] / tau_syn,
            label="Neuron 2",
        )
        axs[1, i].set_xlabel("Epoch")
        if i == 0:
            axs[1, i].set_ylabel(r"Spike time $t [\tau_s]$")
        axs[1, i].legend()


# def plt_spikes(
#     axs, spikes: Spike, t_max: float, observe: Array, target: Optional[Array] = None
# ):
#     for i, it in enumerate(observe):
#         spike_times = spikes.time[it] / t_max
#         s = 3 * (120.0 / len(spikes.time[it])) ** 2.0
#         axs[i].scatter(x=spike_times, y=spikes.idx[it] + 1, s=s, marker="|", c="black")
#         axs[i].set_ylabel("neuron id")
#         axs[i].set_xlabel(r"$t$ [us]")
#         if target is not None:
#             axs[i].scatter(x=target[0] / t_max, y=1, s=s, marker="|", c="red")
#             axs[i].scatter(x=target[1] / t_max, y=2, s=s, marker="|", c="red")


def plt_spikes_per_neuron(
    fig, ax, recording, testset, hidden_size, output_size, epochs
):
    correct_class = np.tile(np.argmin(testset[1], axis=-1), (epochs, 1, 1))
    for neuron_ix in range(hidden_size):
        spikes = np.sum(recording[0].idx == neuron_ix, axis=-1)
        n_spikes = np.mean(spikes, axis=(1, 2))
        ax[0].plot(
            np.arange(len(spikes)), n_spikes, label=f"Neuron {neuron_ix}"
        )
        for output in range(output_size):
            n_spikes_class = np.nanmean(
                np.where(correct_class == output, spikes, np.nan), axis=(1, 2)
            )
            ax[output + 1].plot(
                np.arange(len(spikes)),
                n_spikes_class,
                label=f"Neuron {neuron_ix}",
            )

    for i in range(3):
        ax[i].set_xlabel("Epoch")
        ax[i].set_ylabel("# spikes")
    ax[0].set_title("Avg. Number of Spikes")
    for i in range(1, output_size + 1):
        ax[i].set_title(f"Avg. Number of Spikes for class {i}")
    fig.tight_layout()


def plt_weights(fig, ax, weights: List[Weight]):
    shape = weights[0].input.shape
    for i in range(shape[1]):
        for j in range(shape[2]):
            ax[0].plot(
                np.arange(shape[0]), weights[0].input[:, i, j], color=f"C{j}"
            )
            ax[0].set_title("Input weights")

    if isinstance(weights[0], WeightRecurrent):
        shape = weights[0].recurrent.shape
        for i in range(shape[1]):
            for j in range(shape[2]):
                ax[1].plot(
                    np.arange(shape[0]),
                    weights[0].recurrent[:, i, j],
                    color=f"C{j}",
                )
                ax[1].set_title("Recursive weights")
    if len(weights) > 1:
        shape = weights[1].input.shape
        for i in range(shape[1]):
            for j in range(shape[2]):
                ax[2].plot(
                    np.arange(shape[0]),
                    weights[1].input[:, i, j],
                    color=f"C{j}",
                )
                ax[2].set_title("Output weights")

    for i in range(3):
        ax[i].set_xlabel("Epoch")
        ax[i].set_ylabel("Magnitude of weights")
    fig.tight_layout()


def plt_weights_bin(fig, axs, weights: List[Weight]):
    bins = np.arange(-2.0, 2.0, 0.1)

    # plot for first and last epoch
    for j, epoch in enumerate((0, -1)):
        if isinstance(weights[0], WeightRecurrent):
            hidden_size = weights[0].recurrent.shape[-1] - 3
            plot_weights = [
                weights[0].input[epoch, :, :hidden_size].flatten(),
                weights[0]
                .recurrent[epoch, :hidden_size, hidden_size: hidden_size + 3]
                .flatten(),
            ]
        else:
            plot_weights = [
                weights[0].input[epoch, :, :].flatten(),
                weights[1].input[epoch, :, :].flatten(),
            ]

        for i, param in enumerate(plot_weights):
            axs[j, i].hist(
                param,
                bins=bins,
                rwidth=0.8,
                color="orange",
                histtype="bar",
            )
            axs[j, i].set_xlabel("Weight")
            axs[j, i].set_ylabel("Occurence")
            axs[j, i].title.set_text(f"Layer {i}")
    fig.tight_layout()


def plt_spike_time_bins(
    axs: List[plt.Axes], dataset: Dataset, t_spike: Array, tau_syn: float
):
    t_late = 2.0
    t_spike = np.where(t_spike == np.inf, t_late, t_spike / tau_syn)

    # reshape
    n_classes = dataset[1].shape[-1]
    n_epochs = t_spike.shape[0]
    colors = ("orange", "grey")
    title = ("Before training", "After 10 epochs", f"After {n_epochs} epochs")
    label = ("Correct Neuron", "Wrong Neuron")
    bins = (
        np.arange(0.4, 0.8, 0.02),
        np.arange(0.4, 1.6, 0.02),
        np.arange(0.4, 1.6, 0.02),
    )

    truth = np.argmin(dataset[1].reshape(-1, n_classes), axis=1)
    if dataset[1].shape[-1] == 2:
        not_truth = np.array([1, 0])[truth]
    elif dataset[1].shape[-1] == 3:
        not_truth = np.array([[1, 2], [0, 2], [0, 1]])[truth]
    else:
        raise AssertionError

    for i, epoch in enumerate([0, min(9, n_epochs - 1), n_epochs - 1]):
        # plot two vertical lines
        # plt.axvline(x=correct_class_target, color="blue", label="Target correct label")
        # plt.axvline(x=wrong_class_target, color="green", label="Target wrong label")
        # plot last epoch
        spike_times = t_spike[epoch].reshape(-1, n_classes)

        n_samples = spike_times.shape[0]
        spike_times_correct_class = spike_times[np.arange(n_samples), truth]

        # but only take every second one for counts to match to correct class
        if dataset[1].shape[-1] == 3:
            spike_times_wrong_class = spike_times[
                np.arange(n_samples)[:, None], not_truth
            ][::2].flatten()
        else:
            spike_times_wrong_class = spike_times[
                np.arange(n_samples), not_truth
            ]

        combined = [spike_times_correct_class, spike_times_wrong_class]
        axs[i].hist(
            combined,
            bins=bins[i],
            color=colors,
            rwidth=1.0,
            histtype="bar",
            label=label,
        )
        axs[i].set_xlabel(r"t $[\tau_s]$")
        axs[i].set_ylabel("Occurence")
        axs[i].legend()
        # axs[i].set_xticks(np.linspace(0.4, 1.6, 4))
        axs[i].title.set_text(title[i])


def plt_and_save(
    folder,
    testset,
    testresult: TestResult,
    weights_over_time,
    tau_syn: float,
    hidden_size: int,
    epochs: int,
    duplication: int = 1,
    duplicate_neurons: bool = False,
    mock_hw: bool = False,
):
    output_size = testset[1].shape[-1]

    # visualize single sample
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
    input_size = testset[0].time.shape[-1] / (
        1 if duplicate_neurons else duplication
    )
    plt_spikes(
        ax, recording, (input_size, hidden_size, output_size), mock_hw, tau_syn
    )
    fig.tight_layout()
    fig.savefig(f"{folder}/test_sample.png", dpi=300)

    # loss and accuracy
    fig, ax1 = plt.subplots(2, 1, figsize=(6, 10), sharex=True)
    plt_loss(ax1[0], loss)
    plt_accuracy(ax1[1], acc)
    fig.tight_layout()
    plt.xlabel("Epoch")
    fig.subplots_adjust(bottom=0.15)
    fig.savefig(f"{folder}/loss.png", dpi=300)

    # prob spike
    fig, ax1 = plt.subplots(output_size, 1, figsize=(6, 10), sharex=True)
    plt_no_spike_prob(ax1, t_spike, testset)
    fig.tight_layout()
    plt.xlabel("Epoch")
    fig.subplots_adjust(bottom=0.15)
    fig.savefig(f"{folder}/prob_spike.png", dpi=300)

    # 2d spike times
    n_output: int = testset[1].shape[-1]
    fig, ax1 = plt.subplots(1, n_output, figsize=(4 * (n_output + 0.2), 4))
    plt_t_spike_neuron(
        fig, ax1, testset, t_spike, tau_syn, duplication, duplicate_neurons
    )
    fig.tight_layout()
    fig.savefig(f"{folder}/spike_times.png", dpi=150)

    observe = None
    # trajectory
    # observe = ((0, 2, "^"), (0, 5, "s"), (0, 6, "D"))
    # fig, axs = plt.subplots(2, 3, figsize=(10, 7))
    # plt_2dloss(axs, t_spike, testset, observe, tau_syn)
    # fig.tight_layout()
    # fig.savefig(f"{folder}/trajectory.png", dpi=150)

    # classification
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 4))
    plt_dataset(
        axs[0], testset, tau_syn, observe, duplication, duplicate_neurons
    )
    plt_prediction(
        axs[1], testset, t_spike, tau_syn, duplication, duplicate_neurons
    )
    fig.tight_layout()
    fig.savefig(f"{folder}/classification.png", dpi=150)

    # spike time bins
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    plt_spike_time_bins(axs, testset, t_spike, tau_syn)
    fig.tight_layout()
    fig.savefig(f"{folder}/spike_time_bins.png", dpi=150)

    if recording is not None:
        # spikes per neuron
        fig, axs = plt.subplots(output_size + 1, 1, figsize=(7, 7))
        plt_spikes_per_neuron(
            fig, axs, recording, testset, hidden_size, output_size, epochs
        )
        fig.savefig(f"{folder}/spikes_per_neuron.png", dpi=150)

    # weights
    fig, axs = plt.subplots(2, 2, figsize=(7, 7))
    # plt_weights(fig, axs, weights_over_time)
    plt_weights_bin(fig, axs, weights_over_time)
    fig.savefig(f"{folder}/weight_bins.png", dpi=150)
