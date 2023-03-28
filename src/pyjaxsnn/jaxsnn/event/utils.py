from typing import List

import jax.numpy as np
from jaxsnn.base.types import Spike, WeightInput, WeightRecurrent


def bump_weights(
    params: List[WeightInput], recording: List[Spike]
) -> List[WeightInput]:
    min_avg_spike = (0.3, 0.0)
    scalar_bump = 5e-3
    batch_size = recording[0].idx.shape[0]
    for i, (layer_recording, layer_params) in enumerate(zip(recording, params)):
        layer_size = layer_params.input.shape[1]
        spike_count = np.array(
            [
                np.sum(layer_recording.idx == neuron_ix) / batch_size
                for neuron_ix in range(layer_size)
            ]
        )
        bump = (spike_count < min_avg_spike[i]) * scalar_bump
        params[i] = WeightInput(layer_params.input + bump)
    return params


def clip_gradient(grad: List[WeightInput]) -> List[WeightInput]:
    for i in range(len(grad)):
        grad[i] = WeightInput(np.where(np.isnan(grad[i].input), 0.0, grad[i].input))
    return grad


def save_params(params: List[WeightInput], filenames: List[str]):
    # TODO this needs to work with pytrees
    for p, filename in zip(params, filenames):
        np.save(filename, p.input, allow_pickle=True)


def save_params_recurrent(params: WeightRecurrent, folder: str):  
    np.save(f"{folder}/weights_input.npy", params.input, allow_pickle=True)
    np.save(f"{folder}/weights_recurrent.npy", params.recurrent, allow_pickle=True)


def load_params_recurrent(folder: str):
    return WeightRecurrent(
        input=np.load(f"{folder}/weights_input.npy",),
        recurrent=np.load(f"{folder}/weights_recurrent.npy",),
    )


def load_params(filenames) -> List[WeightInput]:
    return [WeightInput(np.load(f)) for f in filenames]


def get_index_trainset(trainset, idx):
    return (Spike(trainset[0].time[idx], trainset[0].idx[idx]), trainset[1][idx])
