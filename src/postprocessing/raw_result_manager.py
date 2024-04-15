import os
import pandas

RAW_RESULT_PATH = "results/raw_results"


def read_raw_result():

    if not os.path.exists(RAW_RESULT_PATH):
        return None

    subjects = os.listdir(RAW_RESULT_PATH)
    raw_results = {}
    for subject in subjects:
        path = RAW_RESULT_PATH + "/" + subject
        representations = os.listdir(path)
        rep_results = {}
        for representation in representations:
            rep_path = path + '/' + representation
            types = os.listdir(rep_path)
            type_result = {}
            for typ in types:
                type_path = rep_path + "/" + typ
                files = os.listdir(type_path)
                file_result = {}
                for file in files:
                    if file.endswith(".csv"):
                        file_path = type_path + "/" + file
                        file_result[file.removesuffix(".csv")] = pandas.read_csv(file_path)
                type_result[typ] = file_result
            rep_results[representation] = type_result
        raw_results[subject] = rep_results

    return raw_results
