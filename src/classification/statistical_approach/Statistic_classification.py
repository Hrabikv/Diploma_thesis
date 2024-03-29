from src.classification.Classifier import Classifier
from src.classification.statistical_approach.Statistic import StatisticalClassification
from timeit import default_timer as timer

l = {"rest": 2, "left": 5, "right": 6, "no_movement": 20, "movement": 30}


class StatisticClassification(Classifier):

    def __init__(self):
        self.classifier = None
        self.name = "StatisticClassification"

    def train(self, data, labels) -> float:
        start = timer()
        classifier = StatisticalClassification(data=data, labels=labels)
        classifier.compute_class_representative()
        classifier.compute_pivots_value_of_representations()
        end = timer()
        time_of_training = end - start

        self.classifier = classifier
        return time_of_training

    def validate(self, data, labels) -> [float, float]:
        start = timer()
        self.classifier.classify(data[0])
        end = timer()
        time_of_classification = end - start
        result_labels = []
        for sample in data:
            result = self.classifier.classify(sample)
            result_labels.append(l.get(result.get_label()))

        correctly = []
        for sample, result in zip(labels, result_labels):
            if sample == result:
                correctly.append(sample)

        if len(labels) == 0:
            return 0, -1

        acc = (len(correctly)/len(labels))*100
        print("Accuracy: {0}".format(acc))

        return acc, time_of_classification
