from jaxsnn.event.hardware.calib import (
    W_66_F3_TAU_MEM_FACTOR_2,
    W_69_F0_TAU_MEM_FACTOR_2,
    W_69_F0_LONG_REFRAC,
    W_63_F3_LONG_REFRAC,
)
import datetime as dt
from jaxsnn.event.hardware.experiment import Experiment
import hxtorch
from jaxsnn.event.hardware.neuron import Neuron
from jaxsnn.event.hardware.input_neuron import InputNeuron
from jaxsnn.event.leaky_integrate_and_fire import LIFParameters
from jaxsnn.base.types import EventPropSpike, WeightInput, Spike
from functools import partial
import jax.numpy as np
from jaxsnn.event.leaky_integrate_and_fire import LIF
from jaxsnn.event.root import ttfs_solver
from jaxsnn.event.hardware.utils import filter_spikes, cut_spikes, linear_saturating
import matplotlib.pyplot as plt

log = hxtorch.logger.get("hxtorch.snn.experiment")

wafer_config = W_69_F0_LONG_REFRAC


def main():
    p = LIFParameters(
        v_reset=-0.0, v_th=1.0, tau_syn_inv=1 / 6e-6, tau_mem_inv=1 / 12e-6
    )
    runtime_us = 50
    weight_scaling = 49
    n_input = 10
    n_spikes = 1
    weights = np.arange(0.3, 1.1, 0.01)
    n_batches = 10
    batch_start_plotting = 0
    n_runs = 10
    offset = 0.3 * 1e-6

    # setup hardware experiment
    experiment = Experiment(calib_path=wafer_config.file, weight_scaling=weight_scaling)
    InputNeuron(n_input, p, experiment)
    Neuron(1, p, experiment)

    inputs = Spike(
        # idx=np.repeat(np.expand_dims(np.arange(n_input), axis=1), n_batches, axis=1).T,
        idx=np.zeros((n_batches, n_input)),
        time=np.zeros((n_batches, n_input)),
    )
    runs = []
    # iterate hardware, three runs
    first_hw_times = []
    first_hw_weight = []
    for run in range(n_runs):
        first_hw_spike = None
        hw_spikes = [[] for _ in range(n_batches)]
        for weight in weights:
            spike = experiment.get_hw_results(
                inputs,
                [WeightInput(np.full((n_input, 1), weight))],
                runtime_us,
                n_spikes=[n_spikes],
            )[0][0]
            for i in range(n_batches):
                hw_spikes[i].append(spike.time[i, 0])
            if not np.isinf(spike.time[0, 0]) and first_hw_spike is None:
                first_hw_spike = 1
                int_weight = round(weight * weight_scaling)
                first_hw_times.append(spike.time[0, 0])
                first_hw_weight.append(int_weight)
                log.INFO(
                    f"One run: First HW spike at weight: {int_weight} and time {spike.time[0, 0]}"
                )
        runs.append(hw_spikes)

    first_hw_spike = (
        np.mean(np.array(first_hw_times)),
        np.mean(np.array(first_hw_weight)),
    )
    log.INFO(
        f"Avg run: First HW spike at weight: {int(first_hw_spike[1])} and time {first_hw_spike[0]}"
    )

    # setup software experiment
    solver = partial(ttfs_solver, p.tau_mem, p.v_th)
    _, apply_fn = LIF(
        n_hidden=1,
        n_spikes=n_input + n_spikes,
        t_max=runtime_us * 1e-6,
        p=p,
        solver=solver,
    )

    # iterate software
    inputs = EventPropSpike(
        idx=np.zeros(n_input, dtype=int),
        time=np.zeros(n_input),
        current=np.zeros(n_input),
    )

    first_sw_spike = None
    sw_spikes = []
    for weight in weights:
        spike = cut_spikes(
            filter_spikes(
                apply_fn(
                    layer_start=1,
                    weights=WeightInput(np.array([weight])),
                    input_spikes=inputs,
                ),
                layer_start=1,
            ),
            n_spikes,
        )
        if not np.isinf(spike.time[0]) and first_sw_spike is None:
            first_sw_spike = (spike.time[0], weight)
            log.INFO(f"First SW spike at weight: {weight:.2f} and time {spike.time[0]}")
        sw_spikes.append(spike.time[0])

    log.INFO(
        f"Suggesting weight mapping factor of {int(first_hw_spike[1] / first_sw_spike[1])}"
    )
    log.INFO(f"Suggesting time shift of {first_hw_spike[0] - first_sw_spike[0]}")

    int_weights = weights * weight_scaling

    # plot
    fig, axs = plt.subplots(1, 1, figsize=(15, 8))
    for i in range(batch_start_plotting, n_batches):
        sum = np.zeros_like(weights)
        # iterate all runs
        for run in range(n_runs):
            data = np.array(runs[run][i]) * 1e6 * 125
            sum += data
            axs.plot(int_weights, data, linewidth=0.2, color="gray")
        # plot mean
        axs.plot(
            int_weights,
            sum / n_runs,
            linewidth=0.5,
            label=f"Batch entry {i}",
        )

    axs.set_title(
        f"Spike time for {n_input} input spikes at t=0, wafer {wafer_config.name}"
    )
    axs.plot(int_weights, (np.array(sw_spikes) + offset) * 1e6 * 125, label="Software")
    axs.set_xlabel(f"Input weight on hardware, scaling factor: {weight_scaling}")
    axs.set_ylabel(r"Spike time in FPGA cycles")  # $[\tau_s]$")
    fig.legend()

    dt_string = dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    fig.savefig(f"jaxsnn/plots/hardware/spike_times/{dt_string}_spike_times.png")
    log.INFO("Saved plot")


if __name__ == "__main__":
    hxtorch.init_hardware()
    main()
    hxtorch.release_hardware()
