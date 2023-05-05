from typing import Optional

import jax.numpy as np
import matplotlib.pyplot as plt
from jax import random, vmap


def get_class(coords):
    return np.where(coords[0] < coords[1], 1, 0)


get_class_batched = vmap(get_class)


class LinearDataset:
    def __init__(
        self,
        key: random.KeyArray,
        size: int = 1000,
    ):
        """
        Initializing the dataset:

        .. code: python
            from jaxsnn.dataset.yinyang import LinearDataset

            dataset_train = LinearDataset(size=5000, key=42)
            dataset_validation = LinearDataset(size=1000, key=41)
            dataset_test = LinearDataset(size=1000, key=40)

        **Note** It is very important to give different seeds for trainings-, validation- and test set, as the data is
        generated randomly using rejection sampling. Therefore giving the same key value will result in having the
        same samples in the different datasets!
        """
        self.vals = np.array([])
        self.classes = np.array([])
        self.class_names = ["inside", "outside"]
        key, subkey = random.split(key)

        coords = random.uniform(key, (size * 2, 2))

        classes = get_class_batched(coords)

        n_per_class = [size // 2, size // 2]
        idx = np.concatenate(
            [np.where(classes == i)[0][:n] for i, n in enumerate(n_per_class)]
        )

        idx = random.permutation(subkey, idx, axis=0)
        self.vals = np.hstack((coords[idx], 1 - coords[idx]))
        self.classes = classes[idx]

    def __getitem__(self, index: int):
        return (self.vals[index], self.classes[index])

    def __len__(self):
        return len(self.classes)


def DataLoader(dataset, batch_size: int, rng: Optional[random.KeyArray]):
    permutation = (
        random.permutation(rng, len(dataset))
        if rng is not None
        else np.arange(len(dataset))
    )
    vals = dataset.vals[permutation].reshape(-1, batch_size, dataset.vals.shape[1])
    classes = dataset.classes[permutation].reshape(-1, batch_size)
    return vals, classes


if __name__ == "__main__":
    rng = random.PRNGKey(42)
    dataset = LinearDataset(rng, size=1000)

    (input, target) = dataset.vals, dataset.classes
    fig, ax = plt.subplots(figsize=(6, 6))
    for color, data in zip(
        ("green", "orange"), (input[target == 0], input[target == 1])
    ):
        ax.scatter(
            data[:, 0],
            data[:, 1],
            c=color,
            s=plt.rcParams["lines.markersize"] ** 2.0 * 2,
            linewidths=1.0,
            edgecolors="black",
            alpha=0.6,
        )

    plt.show()
