from abc import abstractmethod, ABC

import numpy as np


def create_pivots():
    return [Mean(), Median()]


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
        x = np.mean(data, axis=0)
        self.value = x / np.linalg.norm(x)
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
        x = np.median(data, axis=0)
        self.value = x / np.linalg.norm(x)
        return self.value

    def print_name(self):
        return self.name

    def value(self):
        return self.value()
