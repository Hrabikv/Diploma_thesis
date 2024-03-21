from classification.Classifier import Classifier
from classification.statistical_approach.Statistic import StatisticalClassification

l = {"rest": 2, "left": 5, "right": 6}


class StatisticClassification(Classifier):

    def __init__(self):
        self.classifier = None
        self.name = "StatisticClassification"

    def train(self, data, labels):
        classifier = StatisticalClassification(data=data, labels=labels)
        classifier.compute_class_representative()
        classifier.compute_pivots_value_of_representations()
        classifier.chose_best_pivots_globally()

        self.classifier = classifier

    def validate(self, data, labels) -> float:
        result_labels = []
        for sample in data:
            result = self.classifier.classify(sample)
            result_labels.append(l.get(result.get_label()))

        correctly = []
        for sample, result in zip(labels, result_labels):
            if sample == result:
                correctly.append(sample)

        acc = (len(correctly)/len(labels))*100
        print(acc)

        return acc
