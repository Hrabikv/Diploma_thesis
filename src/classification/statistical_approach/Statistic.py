import numpy as np

from .Pivots import create_pivots
from .Metrics import EuclideanDistance
from .Representative import Representative

metrics = [EuclideanDistance()]
l = ["rest", "left", "right"]


class StatisticalClassification:

    def __init__(self, data, labels):
        self.results = None
        self.pivots = None
        self.class_data = None
        self.pom1 = None
        self.pom2 = None
        self._create_class_dependent_data(data=data, labels=labels)

    def _create_class_dependent_data(self, number_of_classes=3, data=None, labels=None):
        self.pom1 = np.array(labels)
        self.pom2 = []
        new_data = []
        for classes in range(number_of_classes):
            new_data.append([])

        for sample, label in zip(data, labels):

            self.pom2.append(np.array([sample[:1000], sample[1000:2000], sample[2000:]]))
            if label == 2:
                new_data[0].append(sample)
            elif label == 5:
                new_data[1].append(sample)
            elif label == 6:
                new_data[2].append(sample)

        self.class_data = new_data

    def compute_class_representative(self):
        class_pivots = []
        for data, label in zip(self.class_data, l):
            representatives = []
            pivots = create_pivots()
            for pivot in pivots:
                pivot.compute(data)
                # plot_data_sample(pivot.compute(data), label)
                # self.pom1 = np.append(self.pom1, 10)
                # self.pom2.append(np.array([pivot.value[:1000], pivot.value[1000:2000], pivot.value[2000:]]))
                # plot_data_of_all_subject(np.array(self.pom2), self.pom1, "{0} + {1}".format(label, pivot.print_name()))
                representatives.append(pivot)

            class_pivots.append(representatives)

        self.pivots = class_pivots

    def compute_pivots_value_of_representations(self):
        results = []
        for pivot, label, data in zip(self.pivots, l, self.class_data):
            for p in pivot:
                for metric in metrics:
                    distance = 0
                    for sample in data:
                        distance = distance + metric.compute_distance(p.value, sample)
                    results.append(Representative(value=distance, metric=metric, pivot=p, label=label))
        self.results = results
        self.chose_best_pivots_globally()

        # self.pom1 = np.concatenate([self.pom1, [1, 4, 7]])
        # for rep in self.results:
        #     self.pom2.append(np.array([rep.get_pivot().value[:1000], rep.get_pivot().value[1000:2000], rep.get_pivot().value[2000:]]))

        # plot_data_of_all_subject(np.array(self.pom2), self.pom1, "00")

    def chose_best_pivots_globally(self):
        best_rest = Representative(value=1000, metric=None, pivot=None, label=None)
        best_left = Representative(value=1000, metric=None, pivot=None, label=None)
        best_right = Representative(value=1000, metric=None, pivot=None, label=None)
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
            distance = rep.get_metric().compute_distance(rep.get_pivot().value, sample)
            # pom.append(np.array([rep.get_pivot().value[:1000], rep.get_pivot().value[1000:2000], rep.get_pivot().value[2000:]]))

            if distance < value:
                result = rep
                value = distance

        # plot_data_of_all_subject(np.array(pom), pom2, "tt")

        return result
