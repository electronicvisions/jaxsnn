import time
from typing import Any, Callable, List, Tuple

import jax.numpy as np
from jaxsnn.event.types import Spike, Weight, WeightInput, WeightRecurrent


def bump_weights(
    weights: List[WeightInput], recording: List[Spike]
) -> List[WeightInput]:
    min_avg_spike = (0.3, 0.0)
    scalar_bump = 5e-3
    batch_size = recording[0].idx.shape[0]
    for i, (layer_recording, layer_weights) in enumerate(
        zip(recording, weights)
    ):
        layer_size = layer_weights.input.shape[1]
        spike_count = np.array(
            [
                np.sum(layer_recording.idx == neuron_ix) / batch_size
                for neuron_ix in range(layer_size)
            ]
        )
        bump = (spike_count < min_avg_spike[i]) * scalar_bump
        weights[i] = WeightInput(layer_weights.input + bump)
    return weights


def clip_gradient(grad: List[WeightInput]) -> List[WeightInput]:
    for i in range(len(grad)):
        grad[i] = WeightInput(
            np.where(np.isnan(grad[i].input), 0.0, grad[i].input)
        )
    return grad


def save_weights(weights: List[Weight], folder: str):
    for i, p in enumerate(weights):
        filename = f"{folder}/weights_{i}.npy"
        np.save(filename, p.input, allow_pickle=True)


def save_weights_recurrent(weights: WeightRecurrent, folder: str):
    np.save(f"{folder}/weights_input.npy", weights.input, allow_pickle=True)
    np.save(
        f"{folder}/weights_recurrent.npy", weights.recurrent, allow_pickle=True
    )


def load_weights_recurrent(folder: str):
    return WeightRecurrent(
        input=np.load(
            f"{folder}/weights_input.npy",
        ),
        recurrent=np.load(
            f"{folder}/weights_recurrent.npy",
        ),
    )


def load_weights(filenames) -> List[WeightInput]:
    return [WeightInput(np.load(f)) for f in filenames]


def get_index_trainset(trainset, idx):
    return (
        Spike(trainset[0].time[idx], trainset[0].idx[idx]),
        trainset[1][idx],
    )


def time_it(fn: Callable, *args) -> Tuple[Any, float]:
    start = time.time()
    res = fn(*args)
    return res, time.time() - start
