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
        types = os.listdir(path)
        type_result = {}
        for typ in types:
            type_path = path + "/" + typ
            files = os.listdir(type_path)
            file_result = {}
            for file in files:
                if file.endswith(".csv"):
                    file_path = type_path + "/" + file
                    file_result[file.removesuffix(".csv")] = pandas.read_csv(file_path)
            type_result[typ] = file_result
        raw_results[subject] = type_result

    return raw_results
