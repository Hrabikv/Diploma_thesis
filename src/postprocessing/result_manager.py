from .raw_result_manager import read_raw_result
from src.vizualization.Results_vizualization import plot_results
import numpy as np
import pandas
import os


def compute_final_results():
    raw_results = read_raw_result()
    if raw_results is None:
        return None
    results = {}
    names = ["average_accuracy", "best_accuracy", "time_of_classification", "time_of_training"]
    for subject in raw_results:
        sub = raw_results[subject]
        types_results = {}
        for typ in sub:
            t = sub[typ]
            class_results = {}
            for classifier in t:
                clas = t[classifier]
                values = [np.mean(clas["accuracy"]), np.max(clas["accuracy"]), np.mean(clas["time_of_classification"]), np.mean(clas["time_of_training"])]
                tmp_results = {"names": names,
                               "values": values}
                class_results[classifier] = tmp_results
            types_results[typ] = class_results
        results[subject] = types_results

    return results


def save_results(data_per_class, path):
    # if os.path.exists(path):
    #     os.remove(path)
    df = pandas.DataFrame(data_per_class)
    df.to_csv(path)


def create_final_results():
    results = compute_final_results()
    if results is None:
        return
    path = "results"
    for subject in results:
        subject_path = path + "/" + subject
        if not os.path.exists(subject_path):
            os.mkdir(subject_path)
        sub = results[subject]
        for typ in sub:
            type_path = subject_path + "/" + typ
            if not os.path.exists(type_path):
                os.mkdir(type_path)
            t = sub[typ]
            save_results(t, type_path + "/results.csv")
            plot_results(t, type_path)
    print()
