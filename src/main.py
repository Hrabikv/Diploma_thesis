from preprocessing import ETL
from classification.Cross_Validation import cross_validation
import numpy as np
import keras.layers as layers
import keras
from classification.statistical_approach.Statistic_classification import StatisticClassification
from classification.cnn.CNN import CNN
from classification.mlp.MLP import MLP
from classification.lstm.LSTM import LSTM
from classification.transformer.Transformer import Transformer


def tests_all_subjects(data, labels, classifiers):
    i = 1
    for subject, label in zip(data, labels):
        cross_validation(vectors=subject, labels=label, classifiers=classifiers, subject="{}".format(i))
        i = i + 1


if __name__ == '__main__':
    d, l = ETL.load_data()
    classifiers_array = [Transformer()]
    # cross_validation(vectors=d[8], labels=l[8], classifiers=classifiers_array, subject="9")
    tests_all_subjects(d, l, classifiers_array)
    print()

