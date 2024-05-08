import sys

import numpy as np

from classification.Cross_Validation import cross_validation
from classification.statistical_approach.Statistic_classification import StatisticClassification
from classification.cnn.CNN import CNN
from classification.mlp.MLP import MLP
from classification.lstm.LSTM import LSTM
from classification.transformer.Transformer import Transformer
from postprocessing.result_manager import create_final_results
from preprocessing import ETL
from utils.config import Config
from utils.configReader import readConfig


def test_all_subjects(data, labels, classifiers):
    i = 1
    for subject, label in zip(data, labels):
        name = i
        if i < 10:
            name = '0{}'.format(i)
        cross_validation(vectors=subject, labels=label, classifiers=classifiers, subject="subject_{}".format(name))
        i += 1


def test_whole_dataset(data, labels, classifiers):
    data = np.concatenate(data)
    labels = np.concatenate(labels)
    cross_validation(vectors=data, labels=labels, classifiers=classifiers, subject="all")


def test_Kodera_29(data, labels, classifiers):
    data = np.concatenate(np.array(data)[:29])
    labels = np.concatenate(np.array(labels)[:29])
    cross_validation(vectors=data, labels=labels, classifiers=classifiers, subject="Kodera_29")


def test_Farabbi_12(data, labels, classifiers):
    data = np.concatenate((np.array(data)[29:]))
    labels = np.concatenate((np.array(labels)[29:]))
    cross_validation(vectors=data, labels=labels, classifiers=classifiers, subject="Farabbi_12")


def create_array_of_classifiers():
    classifiers = {"Statistic": StatisticClassification(),
                   "MLP": MLP(),
                   "CNN": CNN(),
                   "LSTM": LSTM(),
                   "Transformer": Transformer()}
    result = []
    for clas in Config.CLASSIFIERS:
        result.append(classifiers[clas])
    return result


if __name__ == '__main__':
    try:
        readConfig()
    except Exception as error:
        print('Caught this error: ' + repr(error))
        sys.exit(-1)

    d, l = ETL.load_data()
    classifiers_array = create_array_of_classifiers()

    if Config.TYPE_OF_DATA == "intra-subject":
        test_all_subjects(d, l, classifiers_array)
    if Config.TYPE_OF_DATA == "all":
        test_whole_dataset(d, l, classifiers_array)
    if Config.TYPE_OF_DATA == "Kodera_29":
        test_Kodera_29(d, l, classifiers_array)
    if Config.TYPE_OF_DATA == "Farabbi_12":
        test_Farabbi_12(d, l, classifiers_array)

    create_final_results()

