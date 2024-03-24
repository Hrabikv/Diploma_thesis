from abc import ABC, abstractmethod


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
