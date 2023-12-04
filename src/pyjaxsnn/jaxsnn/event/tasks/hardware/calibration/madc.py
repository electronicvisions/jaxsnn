"""Plot the expected spike time of a single neuron with multiple input
spikes in software against the measured MADC trace. This can be used
to find the proper weight scaling and cycle offset factor.
"""
import datetime as dt
import logging

import hxtorch
import jax.numpy as np
import matplotlib.pyplot as plt
import numpy as onp
from dlens_vx_v3 import hal
from jaxsnn.event.hardware.calib import W_69_F0_LONG_REFRAC
from jaxsnn.event.hardware.experiment import Experiment
from jaxsnn.event.hardware.input_neuron import InputNeuron
from jaxsnn.event.hardware.neuron import Neuron
from jaxsnn.event.hardware.utils import (
    cut_spikes,
    filter_spikes,
    simulate_madc,
)
from jaxsnn.event.leaky_integrate_and_fire import LIF, LIFParameters
from jaxsnn.event.types import EventPropSpike, Spike, WeightInput

log = logging.getLogger("root")

wafer_config = W_69_F0_LONG_REFRAC
cycles_per_us = int(hal.Timer.Value.fpga_clock_cycles_per_us)


def fill_zeros_with_last(arr):
    prev = np.arange(len(arr))
    prev = prev.at[arr == 0].set(0)
    prev = onp.maximum.accumulate(prev)
    return arr[prev]


def main():
    params = LIFParameters(
        v_reset=-1_000.0, v_th=1.0, tau_syn_inv=1 / 6e-6, tau_mem_inv=1 / 12e-6
    )
    runtime_us = 15
    n_output_spikes = 4
    duplication = 3
    n_input = 2
    input_neurons = n_input * duplication
    weight = 1.0

    HW_CYCLE_CORRECTION = -50
    weights = [WeightInput(input=np.full((input_neurons, 1), weight))]

    inputs = Spike(
        time=np.repeat(
            np.array([200, 500]) / (1e6 * cycles_per_us),
            duplication,
            axis=0,
        ),
        idx=np.arange(input_neurons),
    )

    batched_inputs = Spike(
        time=np.expand_dims(inputs.time, axis=0),
        idx=np.expand_dims(inputs.idx, axis=0),
    )

    # setup hardware experiment
    experiment = Experiment(wafer_config)
    InputNeuron(input_neurons, params, experiment)
    Neuron(
        1, params, experiment, enable_madc_recording=True, record_neuron_id=0
    )

    hw_spike, madc_recording = experiment.get_hw_results(
        batched_inputs,
        weights,
        runtime_us,
        n_spikes=[n_output_spikes],
        hw_cycle_correction=HW_CYCLE_CORRECTION,
    )
    hw_spike = hw_spike[0]
    madc_recording = madc_recording[:, :1000]

    # setup software experiment
    _, apply_fn = LIF(
        n_hidden=1,
        n_spikes=n_input * duplication + n_output_spikes,
        t_max=runtime_us * 1e-6,
        params=params,
    )

    # iterate software
    inputs = EventPropSpike(
        idx=inputs.idx,
        time=inputs.time,
        current=np.zeros_like(inputs.time),
    )

    sw_spike = apply_fn(
        layer_start=n_input * duplication,
        weights=weights[0],
        input_spikes=inputs,
    )
    sw_spike = cut_spikes(
        filter_spikes(
            sw_spike,
            layer_start=n_input * duplication,
        ),
        n_output_spikes,
    )

    # plot
    fig, axs = plt.subplots(1, 1, figsize=(15, 8))

    # madc trace
    len = madc_recording.shape[0]

    log.info(f"Simulating madc with len {len}")
    sw_madc = simulate_madc(
        params.tau_mem_inv,
        params.tau_syn_inv,
        inputs,
        weight,
        np.arange(len) / (1e6 * cycles_per_us),
    )
    sw_madc = sw_madc[:, 0] * 190 + 315
    prettified = fill_zeros_with_last(madc_recording[:, 0])

    # find first non zero
    first_non_zero = np.argmax(prettified != 0)
    prettified[:first_non_zero] = prettified[first_non_zero]

    hw_spike_time = hw_spike.time[0, 0] * 1e6 * cycles_per_us
    sw_spike_time = sw_spike.time[0] * 1e6 * cycles_per_us
    log.info(f"SW spike time: {sw_spike_time}")

    log.info(f"HW spike time: {hw_spike_time}")

    # first batch
    axs.set_title(
        f"MADC recording, wafer {wafer_config.name}, "
        "input spike after 200 and 500 cycles, "
        f"hw cycle correction: {HW_CYCLE_CORRECTION}, "
        f"weight scaling: {wafer_config.weight_scaling}"
    )
    axs.plot(np.arange(len) + HW_CYCLE_CORRECTION, prettified)
    axs.plot(np.arange(int(sw_spike_time)), sw_madc[: int(sw_spike_time)])
    axs.set_xlabel(f"FPGA Clock cycles")
    axs.set_ylabel("MADC value")
    fig.legend()

    dt_string = dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    fig.savefig(f"data/event/hardware/madc/{dt_string}_spike_times.png")
    np.save(
        f"data/event/hardware/madc/{dt_string}_trace.npy",
        madc_recording,
        allow_pickle=True,
    )
    log.info(
        f"Count: {len}, "
        f"Non zero count: {np.count_nonzero(madc_recording[:, 0])}"
    )


if __name__ == "__main__":
    hxtorch.init_hardware()
    main()
    hxtorch.release_hardware()
