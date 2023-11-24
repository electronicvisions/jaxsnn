# pylint: disable=invalid-name
from typing import Optional

import jax
import jax.numpy as np
from jax import random


def outside_circle(x: float, y: float, r_big) -> bool:
    return np.sqrt((x - r_big) ** 2 + (y - r_big) ** 2) >= r_big


def dist_to_right_dot(x: int, y: int, r_big) -> float:
    return np.sqrt((x - 1.5 * r_big) ** 2 + (y - r_big) ** 2)


def dist_to_left_dot(x: int, y: int, r_big) -> float:
    return np.sqrt((x - 0.5 * r_big) ** 2 + (y - r_big) ** 2)


def get_class(coords, r_big: float, r_small: float):
    # # equations inspired by
    # # https://link.springer.com/content/pdf/10.1007/11564126_19.pdf
    # outside of circle is a different class
    x, y = coords
    d_right = dist_to_right_dot(x, y, r_big)
    d_left = dist_to_left_dot(x, y, r_big)
    criterion1 = d_right <= r_small
    criterion2 = np.logical_and(d_left > r_small, d_left <= 0.5 * r_big)
    criterion3 = np.logical_and(y > r_big, d_right > 0.5 * r_big)
    is_yin = np.logical_or(np.logical_or(criterion1, criterion2), criterion3)
    is_circles = np.logical_or(d_right < r_small, d_left < r_small)
    return (
        is_circles.astype(int) * 2
        + np.invert(is_circles).astype(int) * is_yin.astype(int)
        + outside_circle(x, y, r_big) * 10
    )


get_class_batched = jax.vmap(get_class, in_axes=(0, None, None))


# TODO #4038: Feels like duplicating hxtorch functionality
class YinYangDataset:
    def __init__(  # pylint: disable=too-many-arguments
        self,
        rng: random.KeyArray,
        size: int = 1000,
        r_small: float = 0.1,
        r_big: float = 0.5,
        bias_spike: Optional[float] = None,
    ):
        """
        Initializing the dataset:

        .. code: python
            from jaxsnn.discrete.dataset.yinyang import YinYangDataset

            dataset_train = YinYangDataset(size=5000, rng=42)
            dataset_validation = YinYangDataset(size=1000, rng=41)
            dataset_test = YinYangDataset(size=1000, rng=40)

        It is very important to give different seeds for trainings-,
        validation- and test set, as the data is generated randomly
        using rejection sampling. Therefore giving the same rng value will
        result in having the same samples in the different datasets.
        """
        self.r_small = r_small
        self.r_big = r_big
        self.vals: jax.Array = np.array([])
        self.classes: jax.Array = np.array([])
        self.class_names = ["yin", "yang", "dot"]
        rng, subkey = random.split(rng)

        # on average we need around 7 tries for one sample
        coords = random.uniform(rng, (size * 10 + 100, 2)) * 2.0 * self.r_big

        classes = get_class_batched(coords, self.r_big, self.r_small)

        n_per_class = [size // 3, size // 3, size - 2 * size // 3]
        idx = np.concatenate(
            [np.where(classes == i)[0][:n] for i, n in enumerate(n_per_class)]
        )

        idx = random.permutation(subkey, idx, axis=0)
        self.vals = np.hstack((coords[idx], 1 - coords[idx]))
        if bias_spike is not None:
            bias = np.full((len(self.vals), 1), bias_spike)
            self.vals = np.hstack((self.vals, bias))

        self.classes = classes[idx]

    def __getitem__(self, index: int):
        return (self.vals[index], self.classes[index])

    def __len__(self):
        return len(self.classes)


def data_loader(dataset, batch_size: int, rng: Optional[random.KeyArray]):
    permutation = (
        random.permutation(rng, len(dataset))
        if rng is not None
        else np.arange(len(dataset))
    )
    vals = dataset.vals[permutation].reshape(
        -1, batch_size, dataset.vals.shape[1]
    )
    classes = dataset.classes[permutation].reshape(-1, batch_size)
    return vals, classes
