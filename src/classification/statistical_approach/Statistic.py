import sys
import numpy as np

from .Pivots import create_pivots
from .Metrics import EuclideanDistance
from .Representative import Representative
from utils.config import Config

metrics = [EuclideanDistance()]
l3 = ["rest", "left", "right"]
l2 = ["no_movement", "movement"]


class StatisticalClassification:

    def __init__(self, data, labels):
        self.results = None
        self.pivots = None
        self.class_data = None
        self.pom1 = None
        self.pom2 = None
        self.labels = None
        self.labels_names = None
        if Config.NUMBER_OF_CLASSES == 2:
            self.labels = l2
        elif Config.NUMBER_OF_CLASSES == 3:
            self.labels = l3
        self._create_class_dependent_data(data=data, labels=labels, number_of_classes=Config.NUMBER_OF_CLASSES)

    def _create_class_dependent_data(self, number_of_classes=Config.NUMBER_OF_CLASSES, data=None, labels=None):
        self.pom1 = np.array(labels)
        self.pom2 = []
        new_data = []
        for classes in range(number_of_classes):
            new_data.append([])

        self.size = int(len(data[0])/3)

        for sample, label in zip(data, labels):

            self.pom2.append(np.array([sample[:self.size], sample[self.size:self.size*2], sample[self.size*2:]]))
            if Config.NUMBER_OF_CLASSES == 2:
                if label == 20:
                    new_data[0].append(sample)
                elif label == 30:
                    new_data[1].append(sample)
            elif Config.NUMBER_OF_CLASSES == 3:
                if label == 2:
                    new_data[0].append(sample)
                elif label == 5:
                    new_data[1].append(sample)
                elif label == 6:
                    new_data[2].append(sample)

        self.class_data = new_data

    def compute_class_representative(self):
        class_pivots = []
        for data, label in zip(self.class_data, self.labels):
            representatives = []
            if len(data) == 0:
                continue
            pivots = create_pivots()
            for pivot in pivots:
                pivot.compute(data)
                representatives.append(pivot)

            class_pivots.append(representatives)

        self.pivots = class_pivots

    def compute_pivots_value_of_representations(self):
        results = []
        for pivot, label, data in zip(self.pivots, self.labels, self.class_data):
            for p in pivot:
                for metric in metrics:
                    distance = 0
                    for sample in data:
                        distance = distance + metric.compute_distance(p.value, sample)
                    results.append(Representative(value=distance, metric=metric, pivot=p, label=label))
        self.results = results
        if Config.NUMBER_OF_CLASSES == 2:
            self.chose_best_pivots_globally_binary()
        elif Config.NUMBER_OF_CLASSES == 3:
            self.chose_best_pivots_globally_multiclass()
        #
        # self.pom1 = np.concatenate([self.pom1, [8, 8, 8]])
        # for rep in self.results:
        #     self.pom2.append(np.array([rep.get_pivot().value[:self.size], rep.get_pivot().value[self.size:self.size*2],
        #                                rep.get_pivot().value[self.size*2:]]))
        #
        # plot_data_of_all_subject(np.array(self.pom2), self.pom1, "Pivot example")

    def chose_best_pivots_globally_binary(self):
        best_movement = Representative(value=sys.maxsize, metric=None, pivot=None, label=None)
        best_no_movement = Representative(value=sys.maxsize, metric=None, pivot=None, label=None)
        for result in self.results:
            if result.get_label() == "no_movement":
                if result.get_value() < best_no_movement.get_value():
                    best_no_movement = result
            elif result.get_label() == "movement":
                if result.get_value() < best_movement.get_value():
                    best_movement = result

        self.results = [best_movement, best_no_movement]

    def chose_best_pivots_globally_multiclass(self):
        best_rest = Representative(value=sys.maxsize, metric=None, pivot=None, label=None)
        best_left = Representative(value=sys.maxsize, metric=None, pivot=None, label=None)
        best_right = Representative(value=sys.maxsize, metric=None, pivot=None, label=None)
        for result in self.results:
            if result.get_label() == "rest":
                if result.get_value() < best_rest.get_value():
                    best_rest = result
            elif result.get_label() == "left":
                if result.get_value() < best_left.get_value():
                    best_left = result
            elif result.get_label() == "right":
                if result.get_value() < best_right.get_value():
                    best_right = result

        self.results = [best_rest, best_left, best_right]

    def classify(self, sample):
        result = None
        value = 1000
        # pom = []
        # pom2 = []
        # pom2.append(8)
        # pom2.append(2)
        # pom2.append(5)
        # pom2.append(6)
        # pom.append(np.array([sample[:1000], sample[1000:2000], sample[2000:]]))

        for rep in self.results:
            if rep.get_pivot() is None:
                distance = sys.maxsize
            else:
                distance = rep.get_metric().compute_distance(rep.get_pivot().value, sample)
            # pom.append(np.array([rep.get_pivot().value[:1000], rep.get_pivot().value[1000:2000], rep.get_pivot().value[2000:]]))

            if distance < value:
                result = rep
                value = distance

        # plot_data_of_all_subject(np.array(pom), pom2, "tt")

        return result
