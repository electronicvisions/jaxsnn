import unittest
from pathlib import Path

import numpy as np

from jaxsnn.examples.event import leak_over_threshold


class LeakOverThresholdExampleTest(unittest.TestCase):
    """ Tests the Leak-over-Threshold example """

    plot_path = Path(__file__).parent.joinpath("plots")

    def setUp(self):
        self.plot_path.mkdir(exist_ok=True)

    def test_no_stimulation(self) -> None:
        train_args = [
            "--seed=0",
            "--t-max=200e-3",
            "--size=100",
            "--tau-mem=1e-2",
            "--tau-syn=5e-3",
            "--w-std=0",
            "--v-reset=0.0",
            "--v-leak=3.0",
            "--v-th=1.0",
            "--plot",
            f"--plot-path={self.plot_path / 'leak_over_threshold.png'}"
        ]

        times = leak_over_threshold.main(
            leak_over_threshold.get_parser().parse_args(train_args)
        )
        # All inter-spike intervals should be equal
        # NOTE: Add zero, to measure distance of first spike too
        diff = np.diff(np.concatenate(([0.], times)))
        self.assertTrue(np.allclose(diff, diff[0]))
        self.assertGreater(times[0], 0.)

    def test_small_stimulation(self) -> None:
        train_args = [
            "--seed=0",
            "--t-max=200e-3",
            "--size=100",
            "--tau-mem=1e-2",
            "--tau-syn=5e-3",
            "--w-std=1e-2",
            "--v-reset=0.0",
            "--v-leak=3.0",
            "--v-th=1.0",
            "--plot",
            f"--plot-path={self.plot_path / 'leak_over_threshold_stim.png'}"
        ]

        times = leak_over_threshold.main(
            leak_over_threshold.get_parser().parse_args(train_args)
        )
        # All inter-spike intervals should be different
        self.assertTrue(len(np.unique(times)) == times.size)
        self.assertGreater(times[0], 0.)


if __name__ == "__main__":
    unittest.main()
