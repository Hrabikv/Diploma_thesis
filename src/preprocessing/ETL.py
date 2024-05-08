from .Data_manager import save_preprocessed_data, load_preprocessed_data
from .datasets_loaders.Kodera_29 import Kodera
from .datasets_loaders.Farabbi_12 import Farabbi
from utils.config import Config

import sys
import numpy as np


def load_data() -> tuple[np.ndarray, np.ndarray]:
    data = []
    labels = []

    preprocessed_data = load_preprocessed_data()
    if preprocessed_data[0] is not None and preprocessed_data[1] is not None:
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
        # plot_data_of_all_subject(d, l)

    data = np.concatenate(data)
    labels = np.concatenate(labels)

    if Config.NUMBER_OF_CLASSES == 2:
        labels = transform_to_binary(labels)

    save_preprocessed_data(data, labels)

    return data, labels


def transform_to_binary(labels):
    new_labels = []
    for subject in labels:
        new_subject_labels = []
        for event in subject:
            if event == 2:
                new_subject_labels.append(20)
            if event == 5 or event == 6:
                new_subject_labels.append(30)
        new_labels.append(np.array(new_subject_labels))
    return np.array(new_labels, dtype=object)


