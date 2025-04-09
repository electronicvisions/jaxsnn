import os
import unittest

import matplotlib.pyplot as plt
import numpy as np

from jaxsnn.base.dataset import shd_dataset


class TestSHDDataset(unittest.TestCase):

    root = os.path.join(os.path.dirname(__file__), "data")
    plots_dir = os.path.join(os.path.dirname(__file__), "plots")
    kwargs = dict(
        train=True,
        download=True,
        t_max=1.0,
        time_scale=1.0,
        channel_start=70,
        channel_modulo=9,
        root=root,
    )

    def setUp(self):
        os.makedirs(self.plots_dir, exist_ok=True)

    def test_shd_dataset_shape(self):
        (data, _), n_spikes = shd_dataset(**self.kwargs)
        self.assertEqual(tuple(data.shape), (8156, 2, n_spikes))
        self.assertLessEqual(data[:, 1].min(), 70)

    def test_shd_dataset_examples(self):
        (data, labels), _ = shd_dataset(**self.kwargs)

        fig, axes = plt.subplots(nrows=3, figsize=(8, 6))
        for ax in axes:
            idx = np.random.randint(data.shape[0])
            times, channels = data[idx, 0], data[idx, 1]
            ax.scatter(times, channels, s=1)
            ax.set_title(f"Label: {int(labels[idx])}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Channel")
            ax.set_xlim(0, 1.0)
            ax.set_ylim(0, 70)

        fig.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "shd_examples.png"))
        plt.close(fig)


if __name__ == '__main__':
    unittest.main()
