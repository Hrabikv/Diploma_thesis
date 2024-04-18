from .raw_result_manager import read_raw_result
from src.visualization.Results_visualization import plot_results
import numpy as np
import pandas
from src.config import TYPE_OF_DATA
import os


def get_classifiers_names(dict_of_classifiers):
    result = []
    for classifiers in dict_of_classifiers:
        result.append(classifiers)
    return result


def compute_metrics(classifier_results):
    acc = 0
    time_clas = 0
    time_train = 0
    i = 0
    for f in classifier_results["fold"]:
        if f == 10:
            acc += classifier_results["accuracy"][i]
            time_clas += classifier_results["time_of_classification"][i]
            time_train += classifier_results["time_of_training"][i]
            i += 1
    values = []
    if i != 0:
        values.append(acc / i)
        values.append(time_clas / i)
        values.append(time_train / i)
    else:
        values.append(0)
        values.append(0)
        values.append(0)
    values.append(np.max(classifier_results["accuracy"]))
    return values


def compute_final_results():
    raw_results = read_raw_result()
    if raw_results is None:
        return None
    results = {}
    metrics = ["average_accuracy", "time_of_classification", "time_of_training", "best_accuracy"]
    for subject in raw_results:
        sub = raw_results[subject]
        for representations in sub:
            representation = sub[representations]
            for typ in representation:
                t = representation[typ]
                names = get_classifiers_names(t)
                for classifier in t:
                    values = compute_metrics(t[classifier])
                    for i, metric in enumerate(metrics):
                        dict_name = representations + '_' + typ + '_' + metric
                        if dict_name not in results.keys():
                            results[dict_name] = {'names': names}
                        if subject not in results[dict_name].keys():
                            results[dict_name][subject] = []
                        results[dict_name][subject].append(values[i])
    return results


def save_results(data_per_class, path, dir):
    # if os.path.exists(path):
    #     os.remove(path)
    index = data_per_class.pop('names')
    df = pandas.DataFrame(data_per_class, index=index).T
    # df.to_csv(path + '.csv')
    if not os.path.exists(dir):
        os.makedirs(dir)
    df.to_excel(path + '.xlsx')


def save_final_results(results):
    path = "results/" + TYPE_OF_DATA
    for result in results:
        save_results(results[result], path + "/" + TYPE_OF_DATA + '_' + result, path)
    # for subject in results:
    #     subject_path = path + "/" + subject
    #     if not os.path.exists(subject_path):
    #         os.mkdir(subject_path)
    #     sub = results[subject]
    #     for typ in sub:
    #         type_path = subject_path + "/" + typ
    #         if not os.path.exists(type_path):
    #             os.mkdir(type_path)
    #         t = sub[typ]
    #         save_results(t, type_path + "/results.csv")
    #         plot_results(t, type_path)


def create_final_results():
    results = compute_final_results()
    if results is None:
        return
    save_final_results(results)
