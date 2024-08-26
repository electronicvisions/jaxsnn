from typing import Optional, Tuple

import jax
import jax.numpy as np
from jax import random


def outside_circle(x_coord: float, y_coord: float, r_big) -> bool:
    return np.sqrt((x_coord - r_big) ** 2 + (y_coord - r_big) ** 2) >= r_big


def dist_to_right_dot(x_coord: int, y_coord: int, r_big) -> float:
    return np.sqrt((x_coord - 1.5 * r_big) ** 2 + (y_coord - r_big) ** 2)


def dist_to_left_dot(x_coord: int, y_coord: int, r_big) -> float:
    return np.sqrt((x_coord - 0.5 * r_big) ** 2 + (y_coord - r_big) ** 2)


def get_class(coords, r_big: float, r_small: float):
    # equations inspired by
    # https://link.springer.com/content/pdf/10.1007/11564126_19.pdf
    # outside of circle is a different class
    x_coord, y_coord = coords
    d_right = dist_to_right_dot(x_coord, y_coord, r_big)
    d_left = dist_to_left_dot(x_coord, y_coord, r_big)
    criterion1 = d_right <= r_small
    criterion2 = np.logical_and(d_left > r_small, d_left <= 0.5 * r_big)
    criterion3 = np.logical_and(y_coord > r_big, d_right > 0.5 * r_big)
    is_yin = np.logical_or(np.logical_or(criterion1, criterion2), criterion3)
    is_circles = np.logical_or(d_right < r_small, d_left < r_small)
    return (
        is_circles.astype(int) * 2
        + np.invert(is_circles).astype(int) * is_yin.astype(int)
        + outside_circle(x_coord, y_coord, r_big) * 10
    )


def yinyang_dataset(
    rng: random.KeyArray,
    size: int,
    mirror: bool,
    bias_spike: Optional[float],
) -> Tuple[jax.Array, jax.Array]:
    rng, subkey = random.split(rng)
    r_big = 0.5
    r_small = 0.1

    # On average around 7 tries needed for one sample
    coords = random.uniform(rng, (size * 10 + 100, 2)) * 2.0 * r_big
    get_class_batched = jax.vmap(get_class, in_axes=(0, None, None))
    classes = get_class_batched(coords, r_big, r_small)

    # Evenly distribute classes
    n_per_class = [size // 3, size // 3, size - 2 * (size // 3)]
    idx = np.concatenate(
        [np.where(classes == i)[0][:n] for i, n in enumerate(n_per_class)]
    )

    idx = random.permutation(subkey, idx, axis=0)
    coords = coords[idx]
    classes = classes[idx]

    if mirror:
        coords = np.hstack((coords, 1 - coords))

    if bias_spike is not None:
        bias = np.full((len(coords), 1), bias_spike)
        coords = np.hstack((coords, bias))

    return coords, classes, "yinyang"
