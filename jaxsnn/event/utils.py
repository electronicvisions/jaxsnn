from typing import List

import jax.numpy as np

from jaxsnn.base.types import Spike, Weight, Array


def bump_weights(params: List[Weight], recording: List[Spike]) -> List[Weight]:
    min_avg_spike = (0.3, 0.0)
    scalar_bump = 5e-3
    batch_size = recording[0].idx.shape[0]
    for i, (layer_recording, layer_params) in enumerate(zip(recording, params)):
        if isinstance(layer_params, tuple):
            # TODO
            pass
        else:
            layer_size = layer_params.shape[1]
            spike_count = np.array(
                [
                    np.sum(layer_recording.idx == neuron_ix) / batch_size
                    for neuron_ix in range(layer_size)
                ]
            )
            bump = (spike_count < min_avg_spike[i]) * scalar_bump
            params[i] = layer_params + bump
    return params


def clip_gradient(grad: list[Weight]) -> tuple[list[Weight], list[float]]:
    count = []
    for i in range(len(grad)):
        if isinstance(grad[i], tuple):
            count.append(np.mean(np.isnan(grad[i][0])) + np.mean(np.isnan(grad[i][1])))
            grad[i] = (
                np.where(np.isnan(grad[i][0]), 0.0, grad[i][0]),
                np.where(np.isnan(grad[i][1]), 0.0, grad[i][1]),
            )
        else:
            count.append(np.mean(np.isnan(grad[i])))
            grad[i] = np.where(np.isnan(grad[i]), 0.0, grad[i])
    return grad, count


def save_params(params: list[Array], filenames: list[str]):
    # TODO this needs to work with pytrees
    for p, filename in zip(params, filenames):
        np.save(filename, p, allow_pickle=True)


def load_params(filenames) -> list[Array]:
    return [np.load(f) for f in filenames]


def get_index_trainset(trainset, idx):
    return (Spike(trainset[0].time[idx], trainset[0].idx[idx]), trainset[1][idx])
