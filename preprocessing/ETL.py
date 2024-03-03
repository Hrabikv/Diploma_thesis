from .datasets_loaders.Kodera_29 import Kodera
from .datasets_loaders.Farabbi_12 import Farabbi
from preprocessing.datasets_loaders.data_formats.File_format import FileFormat
from utils.Data_manager import save_preprocessed_data, load_preprocessed_data

import sys
import mne
import numpy as np
from mne import Epochs
from mne.time_frequency import EpochsTFR


def load_data() -> tuple[np.ndarray, np.ndarray]:
    data = []
    labels = []

    preprocessed_data = load_preprocessed_data()
    if preprocessed_data[0] is not None:
        return preprocessed_data

    datasets = [Kodera(), Farabbi()]
    sampling_frequency = sys.float_info.max
    for dataset in datasets:
        dataset.load_data()
        sampling_frequency = dataset.find_min_sampling_frequency(sampling_frequency)

    for dataset in datasets:
        d, l = dataset.preprocess_data(sampling_frequency=sampling_frequency)
        data.append(d)
        labels.append(l)

    data = np.concatenate(data)
    labels = np.concatenate(labels)

    save_preprocessed_data(data, labels)

    return data, labels


def epochs_to_time_frequency(epochs: Epochs) -> EpochsTFR:
    """
    Converts given epochs object to time frequency domain using Morlet wavelet method.
    The frequency range is set by config.l_freq and config.h_freq, the time domain is decimated by a factor of 4.

    :param epochs: epochs that are to be converted to time frequency domain
    :return: the converted EpochsTFR object instance
    """
    freqs = np.arange(8, 30 + 1)
    n_cycles = freqs / 2
    epochs_tfr = mne.time_frequency.tfr_morlet(epochs, freqs, n_cycles,
                                               use_fft=True, decim=4, average=False, return_itc=False)

    return epochs_tfr

