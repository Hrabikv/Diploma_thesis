from abc import abstractmethod, ABC

import numpy as np
import scipy.spatial.distance as distance


class Metric(ABC):
    @abstractmethod
    def compute_distance(self, pivot, sample):
        pass

    @abstractmethod
    def print_name(self):
        pass


class EuclideanDistance(Metric):
    def __init__(self):
        self.name = "EuclideanDistance"

    def compute_distance(self, pivot, sample):
        return np.linalg.norm(pivot - sample)

    def print_name(self):
        return self.name


class MahalanobisDistance(Metric):
    def __init__(self):
        self.name = "MahalanobisDistance"

    def compute_distance(self, pivot, sample):
        cov = np.corrcoef(pivot, sample)
        icov = np.linalg.inv(cov)
        return distance.mahalanobis(pivot, sample, icov)

    def print_name(self):
        return self.name


class CosineDistance(Metric):

    def __init__(self):
        self.name = "CosineDistance"

    def compute_distance(self, pivot, sample):
        return distance.cosine(pivot, sample)
    
    def print_name(self):
        return self.name


class HammingDistance(Metric):

    def __init__(self):
        self.name = "HammingDistance"

    def compute_distance(self, pivot, sample):
        return distance.hamming(pivot, sample)

    def print_name(self):
        return self.name

