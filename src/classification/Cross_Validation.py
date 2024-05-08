import numpy as np
from .Classifier import Classifier
from visualization.Results_visualization import plot_raw_results
from utils.config import Config
import csv
import os
import warnings
warnings.filterwarnings("error")

cross_start = 10
cross_stop = 96
base_cross_step = 10
base_cross_folds = 10
print_names = ["accuracy", "time_of_training", "time_of_classification", "fold", "number_of_fold"]


def reshape_data(data):
    new_data = []
    for sample in data:
        x = np.array(sample).reshape(3, int(len(sample)/3))
        new_data.append(x)
    return np.array(new_data)


def normalize_data(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        u = np.mean(data, axis=0)
        s = np.std(data, axis=0)
    return (data - u) / s


def concatenate(data):
    new_data = []
    for sample in data:
        x = np.concatenate(sample)
        new_data.append(normalize_data(x))
    return new_data


def split_data_to_classes(data, labels):
    new_data = []
    new_labels = []
    for classes in range(Config.NUMBER_OF_CLASSES):
        new_data.append([])
        new_labels.append([])

    for sample, label in zip(data, labels):

        if label == 2:
            new_data[0].append(sample)
            new_labels[0].append(label)
        elif label == 5:
            new_data[1].append(sample)
            new_labels[1].append(label)
        elif label == 6:
            new_data[2].append(sample)
            new_labels[2].append(label)
        elif label == 20:
            new_data[0].append(sample)
            new_labels[0].append(label)
        elif label == 30:
            new_data[1].append(sample)
            new_labels[1].append(label)

    return new_data, new_labels


def process_parts(part_array):
    tmp = []
    for part in part_array:
        if len(part) > 0:
            tmp.append(part)
    if len(tmp) > 0:
        return np.concatenate(tmp)
    return []


def create_train_test_sorted(data, step, i):
    first_part = []
    middle_part = []
    last_part = []

    for label in data:
        size = int(len(label) * step / 100)
        first_part.append(label[:size * i])
        middle_part.append(label[size * i:size * i + size])
        last_part.append(label[size * i + size:])

    first_part = process_parts(first_part)
    middle_part = process_parts(middle_part)
    last_part = process_parts(last_part)

    if len(first_part) == 0:
        return last_part, middle_part

    if len(last_part) == 0:
        return first_part, middle_part

    return np.concatenate([first_part, last_part]), middle_part


def create_train_test(data, step, i):
    size = int(len(data) * step / 100)

    first_part = data[:size * i]
    middle_part = data[size * i:size * i + size]
    last_part = data[size * i + size:]

    if len(first_part) == 0:
        return last_part, middle_part
    if len(last_part) == 0:
        return first_part, middle_part
    return np.concatenate([first_part, last_part]), middle_part


def compute_results(classifier, vectors, labels, step, i, is_sorted):
    if is_sorted:
        train_vectors, test_vectors = create_train_test_sorted(vectors, step, i)
        train_labels, test_labels = create_train_test_sorted(labels, step, i)
    else:
        train_vectors, test_vectors = create_train_test(vectors, step, i)
        train_labels, test_labels = create_train_test(labels, step, i)

    if len(test_vectors) < 3:
        return None
    time_of_training = classifier.train(train_vectors, train_labels)
    accuracy, time_of_classification = classifier.validate(test_vectors, test_labels)
    return {print_names[0]: accuracy,
            print_names[1]: time_of_training,
            print_names[2]: time_of_classification,
            print_names[3]: step,
            print_names[4]: i + 1}


def cross_validation(vectors, labels, classifiers: [Classifier], subject):
    steps = range(cross_start, cross_stop, base_cross_step)
    vectors = concatenate(vectors)
    print(subject)
    if Config.NUMBER_OF_CLASSES == 3:
        new_vectors, new_labels = split_data_to_classes(vectors, labels)
    for classifier in classifiers:
        results = []
        results_sorted = []
        i = 0
        for step in steps:
            fold = (int(len(vectors) * step / 100))
            for i in range(int(len(vectors) / fold)):
                unsorted_results = compute_results(classifier, vectors, labels, step, i, False)
                if unsorted_results is None:
                    continue

                results.append(unsorted_results)
                if Config.NUMBER_OF_CLASSES == 3:
                    sorted_results = compute_results(classifier, new_vectors, new_labels, step, i, True)
                    if sorted_results is None:
                        continue
                    results_sorted.append(sorted_results)
        save_results(results, classifier.name, subject)
        if Config.NUMBER_OF_CLASSES == 3:
            save_results(results_sorted, "{0}_sorted".format(classifier.name), subject)


def save_results(values_for_print, classificator, subject):
    path = "results"
    if not os.path.isdir(path):
        os.mkdir(path)
    path = path + "/raw_results"
    if not os.path.isdir(path):
        os.mkdir(path)
    path = path + "/{0}".format(subject)
    if not os.path.isdir(path):
        os.mkdir(path)
    path = path + "/" + Config.FEATURE_VECTOR
    if not os.path.isdir(path):
        os.mkdir(path)

    if Config.NUMBER_OF_CLASSES == 2:
        path = path + "/binary"
    elif Config.NUMBER_OF_CLASSES == 3:
        path = path + "/multi_class"
    if not os.path.isdir(path):
        os.mkdir(path)

    with open("{0}/{1}.csv".format(path, classificator), "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=print_names)

        writer.writeheader()
        for line in values_for_print:
            writer.writerow(line)

    plot_raw_results(values_for_print, "{0}/{1}.png".format(path, classificator))

    # print(values_for_print)
