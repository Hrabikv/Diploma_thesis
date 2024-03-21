from abc import abstractmethod, ABC

import numpy as np


class Classifier(ABC):

    name = None

    def train(self, data, labels):
        pass

    def validate(self, data, labels) -> float:
        pass
