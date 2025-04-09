""" SHD Dataset class """
from typing import (
    Tuple,
    Optional,
    Union,
)
import os
import os.path
import gzip
import pathlib
import urllib
import urllib.request
import h5py

import numpy as np
import jax
import jax.numpy as jnp
import jaxsnn

log = jaxsnn.get_logger("jaxsnn.base.dataset.shd")


def shd_dataset(  # pylint: disable=too-many-arguments, too-many-locals
    train: bool,
    download: bool,
    t_max: float,
    root: str,
    time_scale: float = 1.0,
    channel_start: int = 0,
    channel_modulo: int = 1,
    num_samples: Optional[int] = None,
    dtype: np.dtype = np.float32,
) -> Tuple[jax.Array, jax.Array]:
    """
    SHD dataset
    """

    resources = [("https://compneuro.net/datasets/shd_test.h5.gz",
                  "3062a80ec0c5719404d5b02e166543b1"),
                 ("https://compneuro.net/datasets/shd_train.h5.gz",
                  "d47c9825dee33347913e8ce0f2be08b0")]

    _training_file = 'shd_train.h5'
    _test_file = 'shd_test.h5'

    def load_data():
        """
        Download and rescale the SHD data if it doesn't exist in
        processed_folder already
        """
        if not _check_exists():
            if not download:
                raise RuntimeError(
                    "Dataset not found. You can use download=True to download"
                    " it"
                )
            # download files
            os.makedirs(root, exist_ok=True)
            for url, _ in resources:
                filename = url.rpartition('/')[2]
                download_and_extract_archive(
                    url,
                    download_root=root,
                    filename=filename,
                )
        return h5py.File(
            os.path.join(root, _training_file if train else _test_file)
        )

    def _check_exists() -> bool:
        return (
            os.path.exists(os.path.join(root, _training_file))
            and os.path.exists(os.path.join(root, _test_file))
        )

    # get data
    data = load_data()

    if num_samples is None:
        num_samples = len(data["spikes/times"])

    times = np.array(data["spikes/times"])[:num_samples]
    units = np.array(data["spikes/units"])[:num_samples]

    max_unit = np.max([max(u) for u in units])

    # Select times < t_max
    units_cut = [u[t < t_max / time_scale] for t, u in zip(times, units)]
    times_cut = [t[t < t_max / time_scale] for t in times]

    # Select channels >= channel_start
    units_cut_sel = [u[u >= channel_start] for u in units_cut]
    times_cut_sel = [
        t[u >= channel_start] for t, u in zip(times_cut, units_cut)
    ]

    # Select channels % channel_modulo == 0
    channels = np.arange(channel_start, max_unit + 1, channel_modulo)
    units_cut_sel_mod = [
        ((u[np.isin(u, channels)] - channel_start) / channel_modulo)
        for u in units_cut_sel
    ]
    times_cut_sel_mod = [
        t[np.isin(u, channels)] for t, u in zip(times_cut_sel, units_cut_sel)
    ]

    # Max number of events in all samples
    max_length = np.max([sample.size for sample in times_cut_sel_mod])

    padded_times = np.array(
        [
            np.pad(
                row, (0, max_length - len(row)), constant_values=(0, np.inf)
            )
            for row in times_cut_sel_mod
        ]
    )
    padded_units = np.array(
        [
            np.pad(
                row.astype(int), (0, max_length - len(row)),
                constant_values=(0, -1)
            )
            for row in units_cut_sel_mod
        ]
    )

    labels = jnp.array(data["labels"])[:num_samples]
    spikes = jnp.stack(
        [padded_times * time_scale, padded_units]
    ).transpose(1, 0, 2)

    return (spikes.astype(dtype), labels), max_length


def download_and_extract_archive(
    url: str,
    download_root: Union[str, pathlib.Path],
    filename: Union[str, None] = None,
) -> None:
    download_root = os.path.expanduser(download_root)
    os.makedirs(download_root, exist_ok=True)

    if filename is None:
        filename = os.path.basename(url)

    archive_path = os.path.join(download_root, filename)
    extracted_path = archive_path.removesuffix(".gz")

    if os.path.exists(extracted_path):
        return

    log.info(f"Downloading {url}")
    urllib.request.urlretrieve(url, archive_path)

    log.info(f"Extracting {archive_path}")
    with gzip.open(archive_path, "rb") as rf, open(extracted_path, "wb") as wf:  # pylint: disable=invalid-name
        wf.write(rf.read())
    os.remove(archive_path)
