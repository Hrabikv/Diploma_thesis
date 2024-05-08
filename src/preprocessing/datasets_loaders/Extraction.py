from abc import ABC, abstractmethod
import numpy as np
from mne import Epochs

from utils.config import Config


def square(eeg):
    eeg = np.power(eeg, 2)
    return eeg


def transform_data_representation(epochs: Epochs) -> np.ndarray:
    if Config.FEATURE_VECTOR == "time":
        return epochs.get_data()

    elif Config.FEATURE_VECTOR == "freq":
        return epochs.compute_psd(fmin=8, fmax=30).get_data()


class Extraction(ABC):
    data_path = "data/"

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def preprocess_data(self, sampling_frequency):
        pass

    @abstractmethod
    def find_min_sampling_frequency(self, min_sampling_frequency: float) -> int:
        pass
