import unittest
import hxtorch
from jaxsnn.event.hardware.calib import neuron_calib
from jaxsnn.examples.event import yinyang_bss2


class YinYangExampleTest(unittest.TestCase):
    """ Tests the YinYang example implementation """

    def test_training(self) -> None:
        """ Run YinYang training and inference """
        # run calibration
        calib_args = [
            "--wafer=None",
            "--leak=80",
            "--reset=80",
            "--threshold=150",
            "--tau-syn=6e-6",
            "--tau-mem=12e-6",
            "--refractory-time=30e-6",
            "--i-synin-gm=500",
            "--synapse-dac-bias=1000",
            "--calib-dir=calib_files/",
            "--calib-name=yinyang_calib.pbin",
        ]
        neuron_calib.custom_calibrate(
            neuron_calib.get_parser().parse_args(calib_args))

        # run training
        train_args = [
            "--seed=0",
            "--duplicate-neurons",
            "--plot",
            "--testset-size=2944",
            "--trainset-size=4992",
            "--t-late=1.2e-05",
            "--correct-target-time=5.4e-06",
            "--wrong-target-time=6.6e-06",
            "--tau-mem=1.2e-05",
            "--tau-syn=6e-06",
            "--epochs=10",
            "--batch-size=64",
            "--lr=0.005",
            "--lr-decay=0.99",
            "--duplication=5",
            "--hidden-size=100",
            "--hw-correction=-20",
            "--max-runtime=50",
            "--weight-scale=43",
            "--calib-name=jenkins",
            "--calib-path=calib_files/yinyang_calib.pbin",
        ]

        hxtorch.init_hardware()
        accuracy = yinyang_bss2.main(
            yinyang_bss2.get_parser().parse_args(train_args))
        hxtorch.release_hardware()
        self.assertGreater(
            accuracy, 0.9, "Accuracy should be greater than 90%")


if __name__ == "__main__":
    unittest.main()
