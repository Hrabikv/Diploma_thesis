from abc import abstractmethod, ABC
import numpy as np

di = {
    2: [1, 0, 0],
    5: [0, 1, 0],
    6: [0, 0, 1],
    20: [1, 0],
    30: [0, 1]
    }


def convert_labels_to_OneHot(labels):
    result = []
    for label in labels:
        result.append(np.array(di[label]))
    return np.array(result)


class Classifier(ABC):

    name = None

    @abstractmethod
    def train(self, data, labels) -> float:
        pass

    @abstractmethod
    def validate(self, data, labels) -> [float, float]:
        pass
