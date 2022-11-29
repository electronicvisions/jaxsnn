from typing import Optional, List, Tuple
from jaxsnn.base.types import ArrayLike
import numpy as np

from jaxsnn.base.types import Array, Spike
from jaxsnn.event.dataset import Dataset
from matplotlib.pyplot import Axes
import matplotlib.pyplot as plt
import matplotlib as mpl


blue = np.array([[47, 66, 87, 210]]) / 256
red = np.array([[103, 43, 40, 210]]) / 256
black = np.array([[0, 0, 0, 0.3]])
# green = np.array([[47, 75, 37, 210]]) / 256


def plt_loss(ax: Axes, loss: Array):
    ax.plot(np.arange(len(loss)), loss)
    ax.set_ylabel("test loss")


def plt_accuracy(ax: Axes, accuracy: Array):
    ax.plot(np.arange(len(accuracy)), accuracy)
    ax.set_ylabel("test accuracy")


def plt_no_spike_prob(ax: Axes, t_spike_correct, t_spike_false):
    prob_correct = np.mean(t_spike_correct == np.inf, axis=(1, 2))
    prob_false = np.mean(t_spike_false == np.inf, axis=(1, 2))

    ax.plot(np.arange(len(prob_correct)), prob_correct, label="Correct neuron")
    ax.plot(np.arange(len(prob_false)), prob_false, label="False neuron")
    ax.set_ylabel("prob no spike")
    ax.legend()


def plt_average_spike_time(
    ax: Axes, t_spike_correct: Array, t_spike_false: Array, const_target: Array
):
    t_spike_correct[t_spike_correct == np.inf] == np.nan
    t_spike_false[t_spike_false == np.inf] == np.nan

    t_spike_correct_avg = np.nanmean(t_spike_correct, axis=(1, 2))
    t_spike_false_avg = np.nanmean(t_spike_false, axis=(1, 2))

    ax.plot(
        np.arange(len(t_spike_correct_avg)), t_spike_correct_avg, label="Correct neuron"
    )
    ax.axhline(const_target[0], color="red")

    # plot spike time non-correct neuron
    ax.plot(np.arange(len(t_spike_false_avg)), t_spike_false_avg, label="False neuron")
    ax.axhline(const_target[2], color="red")

    ax.set_xlabel("Epoch")
    ax.title.set_text("Output spike times")
    ax.legend()


def plt_prediction(ax: Axes, dataset: Dataset, t_spike: Array, tau_syn_inv: ArrayLike):
    x = dataset[0].time[..., 0].flatten() * tau_syn_inv
    y = dataset[0].time[..., 1].flatten() * tau_syn_inv
    pred_class = np.argmin(t_spike[-1], axis=-1).flatten()
    ax.scatter(
        x[pred_class == 0],
        y[pred_class == 0],
        s=30,
        facecolor=blue,
        edgecolor=black,
        linewidths=0.7,
    )
    ax.scatter(
        x[pred_class == 1],
        y[pred_class == 1],
        s=30,
        facecolor=red,
        edgecolor=black,
        linewidths=0.7,
    )
    ax.plot(np.linspace(0, 1.5, 100), np.linspace(0, 1.5, 100), c="k")
    ax.set_xlabel(r"Input time $t_x [\tau_s]$")
    ax.set_ylabel(r"Input time $t_y [\tau_s]$")
    ax.title.set_text("Classification")


def plt_t_spike_neuron(
    fig, axs: List[Axes], dataset: Dataset, t_spike: Array, tau_syn_inv: ArrayLike
):
    names = ("First", "Second", "Third")
    n_neurons = t_spike[-1].shape[-1]
    x = dataset[0].time[..., 0].flatten() * tau_syn_inv
    y = dataset[0].time[..., 1].flatten() * tau_syn_inv

    t_spike = t_spike[-1].reshape(-1, n_neurons)

    # normalize all neurons with the same value
    cmap = mpl.colormaps["magma"]
    normalize = np.nanpercentile(np.where(t_spike == np.inf, np.nan, t_spike), 95)
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
            x[score == np.inf], y[score == np.inf], facecolor=black, edgecolor=black
        )
        axs[i].plot(np.linspace(0, 1.5, 100), np.linspace(0, 1.5, 100), c="k")
        axs[i].set_xlabel(r"Input time $t_x [\tau_s]$")
        axs[i].set_xticks(np.linspace(0, 1.5, 4))
        axs[i].set_yticks(np.linspace(0, 1.5, 4))
        if i == 0:
            axs[i].set_ylabel(r"Input time $t_y [\tau_s]$")
        axs[i].title.set_text(f"{names[i]} neuron")

    # Normalizer
    norm = mpl.colors.Normalize(vmin=0, vmax=normalize * tau_syn_inv)

    # creating ScalarMappable
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=axs, location="right")
    cbar.ax.set_ylabel(r"$t_{spike}[\tau_s]$", rotation=90, fontsize=13)
    fig.subplots_adjust(bottom=0.15, right=0.7, top=0.9)


def plt_dataset(
    ax: Axes,
    dataset: Dataset,
    observe: List[Tuple[int, int, str]],
    tau_syn_inv: ArrayLike,
):
    x = dataset[0].time[..., 0].flatten() * tau_syn_inv
    y = dataset[0].time[..., 1].flatten() * tau_syn_inv
    label = np.argmin(dataset[1], axis=-1).flatten()
    ax.scatter(
        x[label == 0],
        y[label == 0],
        s=30,
        facecolor=blue,
        edgecolor=black,
        linewidths=0.7,
    )
    ax.scatter(
        x[label == 1],
        y[label == 1],
        s=30,
        facecolor=red,
        edgecolor=black,
        linewidths=0.7,
    )
    for ix1, ix2, marker in observe:
        coords = dataset[0].time[ix1, ix2] * tau_syn_inv
        x, y = coords[0], coords[1]
        ax.scatter(x, y, marker=marker, color="yellow", s=50)
    ax.plot(np.linspace(0, 1.5, 100), np.linspace(0, 1.5, 100), c="k")
    ax.set_xlabel(r"Input time $t_x [\tau_s]$")
    ax.set_ylabel(r"Input time $t_y [\tau_s]$")
    ax.title.set_text("Dataset")


def plt_2dloss(
    axs: Axes,
    t_spike: Array,
    dataset: Array,
    observe: List[Tuple[int, int, str]],
    tau_mem: float,
    tau_syn: float,
):
    for i, (ix1, ix2, marker) in enumerate(observe):
        trajectory = t_spike[:, ix1, ix2, :]
        target = dataset[1][ix1, ix2]

        n = 100

        def loss_fn_vec(t1, t2, target_1, target_2, tau_mem):
            loss_value = -(
                np.log(1 + np.exp(-np.abs(t1 - target_1) / tau_mem))
                + np.log(1 + np.exp(-np.abs(t2 - target_2) / tau_mem))
            )
            return loss_value

        # units of tau syn
        t1 = np.linspace(0, 2, n)
        t2 = np.linspace(0, 2, n)
        xx, yy = np.meshgrid(t1, t2)
        zz = loss_fn_vec(xx * tau_syn, yy * tau_syn, target[0], target[1], tau_mem)

        clipped = np.clip(trajectory, 0, 2 * tau_syn)
        axs[0, i].contourf(t1, t2, zz, levels=500, cmap=mpl.colormaps["magma"])
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
            np.arange(trajectory.shape[0]), trajectory[:, 0] / tau_syn, label="Neuron 1"
        )
        axs[1, i].plot(
            np.arange(trajectory.shape[0]), trajectory[:, 1] / tau_syn, label="Neuron 2"
        )
        axs[1, i].set_xlabel("Epoch")
        if i == 0:
            axs[1, i].set_ylabel(r"Spike time $t [\tau_s]$")
        axs[1, i].legend()


def plt_spikes(
    axs, spikes: Spike, t_max: float, observe: Array, target: Optional[Array] = None
):
    for i, it in enumerate(observe):
        spike_times = spikes.time[it] / t_max
        s = 3 * (120.0 / len(spikes.time[it])) ** 2.0
        axs[i].scatter(x=spike_times, y=spikes.idx[it] + 1, s=s, marker="|", c="black")
        axs[i].set_ylabel("neuron id")
        axs[i].set_xlabel(r"$t$ [us]")
        if target is not None:
            axs[i].scatter(x=target[0] / t_max, y=1, s=s, marker="|", c="red")
            axs[i].scatter(x=target[1] / t_max, y=2, s=s, marker="|", c="red")


def plt_weights():
    pass
    # input_params = params_over_time[0][0].reshape(100, -1)
    # for i in range(input_params.shape[1]):
    #     ax3[0].plot(np.arange(epochs), input_params[:, i])
    #     ax3[0].title.set_text("Input params")

    # recursive_params = params_over_time[0][1].reshape(100, -1)
    # for i in range(recursive_params.shape[1]):
    #     ax3[1].plot(np.arange(epochs), recursive_params[:, i])
    #     ax3[1].title.set_text("Recursive params")

    # output_params = params_over_time[1].reshape(100, -1)
    # for i in range(output_params.shape[1]):
    #     ax3[2].plot(np.arange(epochs), output_params[:, i])
    #     ax3[2].title.set_text("Output params")

    # TODO look at activity, when input, when internal, when output
    # add batching
    # look at loss function
    # look at LI neuron in output layer
    # observe = [0, 30, 60, 90]
    # plot_spikes(ax2, recording[0], t_max: Axes, observe)
    # plot_spikes(ax3, recording[1], t_max: Axes, observe, target=const_target)
