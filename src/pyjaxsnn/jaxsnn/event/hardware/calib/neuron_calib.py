import argparse
import os
from pathlib import Path

import quantities as pq

from dlens_vx_v3 import hxcomm, sta, logger
import calix.spiking
from calix.common.base import StatefulConnection


logger.set_loglevel(logger.get("calix"), logger.LogLevel.INFO)
log = logger.get("jaxsnn.event.hardware.calib.neuron_calib")


def get_parser() -> argparse.ArgumentParser:
    """
    Returns an argument parser with all the options.
    """
    parser = argparse.ArgumentParser(
        description="Calibration for hxtorch EventProp testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--wafer", type=str, default=None, metavar="<wafer and fpga>",
        help="Wafer and FPGA number, default=None."
             "Must be specified or returns error.")
    parser.add_argument("--leak", type=int, default=80)
    parser.add_argument("--reset", type=int, default=80)
    parser.add_argument("--threshold", type=int, default=150)
    parser.add_argument("--tau-syn", type=float, default=6e-6)
    parser.add_argument("--tau-mem", type=float, default=6e-6)
    parser.add_argument("--refractory-time", type=float, default=2e-6)
    parser.add_argument("--i-synin-gm", type=int, default=500)
    parser.add_argument("--synapse-dac-bias", type=int, default=1000)

    parser.add_argument("--calib-dir", type=str, default="calib_files/")
    parser.add_argument("--calib-name", type=str, default=None)
    parser.add_argument(
        "--recreate-calib-files", action="store_true", default=False)
    return parser


def custom_calibrate(args: argparse.Namespace) -> None:

    target_path = Path(args.calib_dir + (
        args.calib_name if args.calib_name is not None else (
            f"calibration_{args.wafer}_"
            + f"leak{args.leak:.0f}_"
            + f"th{args.threshold:.0f}_"
            + f"reset{args.reset:.0f}_"
            + f"taus-{args.tau_syn*1e6:.0f}us_"
            + f"taum-{args.tau_mem*1e6:.0f}us_"
            + f"trefrac-{args.refractory_time*1e6:.1f}us_"
            + f"isyningm-{args.i_synin_gm:.0f}_"
            + f"sdbias-{args.synapse_dac_bias:.0f}.pbin")))
    target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if os.path.exists(target_path) and not args.recreate_calib_files:
            log.INFO(f"Calibration file already exists. Using ({target_path}")
            return

        log.TRACE(f"Test opening file {target_path}")
        with target_path.open(mode="w", encoding="utf-8"):
            pass
        log.INFO("Opened file and got no error... now calibrate")
        with hxcomm.ManagedConnection() as connection:
            # create neuron calib target
            log.TRACE("create neuron calib target")
            target = calix.spiking.neuron.NeuronCalibTarget()
            target.tau_mem = args.tau_mem * 1e6 * pq.us
            target.tau_syn = args.tau_syn * 1e6 * pq.us
            target.i_synin_gm = args.i_synin_gm
            target.synapse_dac_bias = args.synapse_dac_bias
            # leak and reset have to be equal to match software simulation:
            target.leak = args.leak
            target.reset = args.reset

            target.threshold = args.threshold  # default = 125
            target.membrane_capacitance = 63  # default = 63
            target.refractory_time = args.refractory_time * 1e6 * pq.us
            # create spiking calib target with neuron calib target
            sp_target = calix.spiking.SpikingCalibTarget()
            sp_target.neuron_target = target

            # calibrate
            log.TRACE(dir(connection))
            calibration_result = sp_target.calibrate(
                StatefulConnection(connection))
            # calibration_result = calix.calibrate(
            #   target=sp_target, cache_paths=[])
            # dump to binary
            log.INFO("dump to binary")
            builder = sta.PlaybackProgramBuilderDumper()
            calibration_result.apply(builder)

            log.INFO("write to target path")
            with target_path.open(mode="wb") as file:
                file.write(sta.to_portablebinary(builder.done()))

    except FileNotFoundError:
        log.ERROR(f"file {target_path} not found.")


if __name__ == "__main__":
    custom_calibrate(get_parser().parse_args())


def hw_calibration(args):

    I_sg = args.I_syn_gm  # pylint: disable=invalid-name
    calib_path = args.calibration_path + f"I_sg_{I_sg}.pbin"

    if os.path.exists(calib_path) and not args.recreate_calib_files:
        log.INFO(f"Calibration file already exists. Using ({calib_path}")

    else:
        if args.info:
            log.INFO("creating calibration file")
        neuron_params = {"tau_mem": args.tau_mem * pq.s,
                         "tau_syn": args.tau_syn * pq.s,
                         "i_synin_gm": I_sg}

        with hxcomm.ManagedConnection() as connection:
            # create neuron calib target
            target = calix.spiking.neuron.NeuronCalibTarget()
            target.tau_mem = neuron_params["tau_mem"]
            target.tau_syn = neuron_params["tau_syn"]
            target.i_synin_gm = neuron_params["i_synin_gm"]
            # leak and reset have to be equal to match software simulation:
            target.leak = 70
            target.reset = 70

            target.threshold = 180  # default = 125
            target.membrane_capacitance = 63  # default = 63
            target.refractory_time = 1. * pq.us  # default = 2. * pq.us

            # create spiking calib target with neuron calib target
            sp_target = calix.spiking.SpikingCalibTarget()
            sp_target.neuron_target = target

            # calibrate
            calibration_result = sp_target.calibrate(
                StatefulConnection(connection))
            # calibration_result = calix.calibrate(
            #   target=sp_target, cache_paths=[])

            # dump to binary
            builder = sta.PlaybackProgramBuilderDumper()
            calibration_result.apply(builder)

            # save
            calib_path = Path(args.calibration_path + f"I_sg_{I_sg}.pbin")
            with calib_path.open(mode="wb") as file:
                file.write(sta.to_portablebinary(builder.done()))

            calib_path = str(calib_path)
            if args.info:
                log.INFO(f"calibration written to '{calib_path}'")

    return calib_path
