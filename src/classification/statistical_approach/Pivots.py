import warnings
from abc import abstractmethod, ABC

import numpy as np

warnings.filterwarnings("error")


def create_pivots():
    return [Mean(), Median()]


def normalize_data(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        u = np.mean(data, axis=0)
        s = np.std(data, axis=0)
    return (data - u) / s


class Pivot(ABC):
    @abstractmethod
    def compute(self, data):
        pass

    @abstractmethod
    def print_name(self):
        pass

    @abstractmethod
    def value(self):
        pass


class Mean(Pivot):

    def __init__(self):
        self.value = None
        self.name = "Mean"

    def compute(self, data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            x = np.mean(data, axis=0)
        self.value = normalize_data(x)
        return self.value

    def print_name(self):
        return self.name

    def value(self):
        return self.value()


class Median(Pivot):
    def __init__(self):
        self.value = None
        self.name = "Media"

    def compute(self, data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            x = np.median(data, axis=0)
        self.value = normalize_data(x)
        return self.value

    def print_name(self):
        return self.name

    def value(self):
        return self.value()
