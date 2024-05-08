from utils.config import Config
import os

configFile = "./config.txt"


def parserLine(line, parameters):
    parts = line.split(" ")
    if parts[0] == "CLASSIFIERS":
        class_array = []
        for i, _ in enumerate(parts):
            if i > 1:
                class_array.append(parts[i].strip(",\n"))
        parameters[parts[0]] = class_array
    else:
        parameters[parts[0]] = parts[2].strip()


def applyConfig(parameters):
    Config.NUMBER_OF_CLASSES = int(parameters["NUMBER_OF_CLASSES"])
    Config.TRAINING_INFO = int(parameters["TRAINING_INFO"])
    Config.TESTING_INFO = int(parameters["TESTING_INFO"])
    Config.CLASSIFIERS = parameters["CLASSIFIERS"]
    Config.TYPE_OF_DATA = parameters["TYPE_OF_DATA"]
    Config.FEATURE_VECTOR = parameters["FEATURE_VECTOR"]
    if parameters["CUDA_USE"] == "true":
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        Config.CUDA_USE = True
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        Config.CUDA_USE = False


def evaluateConfig(parameters):
    if parameters["NUMBER_OF_CLASSES"] != "2" and parameters["NUMBER_OF_CLASSES"] != "3":
        raise ValueError("Parameter of NUMBER_OF_CLASSES has different values that were expected!!")
    if parameters["TRAINING_INFO"] != "0" and parameters["TRAINING_INFO"] != "1" and parameters["TRAINING_INFO"] != "2":
        raise ValueError("Parameter of TRAINING_INFO has different values that were expected!!")
    if parameters["TESTING_INFO"] != "0" and parameters["TESTING_INFO"] != "1" and parameters["TESTING_INFO"] != "2":
        raise ValueError("Parameter of TESTING_INFO has different values that were expected!!")
    classifiers = {"Statistic": "1",
                   "MLP": "1",
                   "CNN": "1",
                   "LSTM": "1",
                   "Transformer": "1"}
    for clas in parameters["CLASSIFIERS"]:
        if classifiers.get(clas) != "1":
            raise ValueError("Parameter of CLASSIFIERS has different values that were expected!! {0} is not known".format(clas))
    if (parameters["TYPE_OF_DATA"] != "intra-subject" and parameters["TYPE_OF_DATA"] != "all"
            and parameters["TYPE_OF_DATA"] != "Kodera_29" and parameters["TYPE_OF_DATA"] != "Farabbi_12"):
        raise ValueError("Parameter of TYPE_OF_DATA has different values that were expected!!")
    if parameters["FEATURE_VECTOR"] != "time" and parameters["FEATURE_VECTOR"] != "freq":
        raise ValueError("Parameter of FEATURE_VECTOR has different values that were expected!!")
    if parameters["CUDA_USE"] != "true" and parameters["CUDA_USE"] != "false":
        raise ValueError("Parameter of FEATURE_VECTOR has different values that were expected!!")


def readConfig():
    parameters = {}
    f = open(configFile, "r")
    for line in f.readlines():
        if line.startswith("#") or line.startswith(" ") or len(line) < 10:
            continue
        parserLine(line, parameters)
    f.close()
    evaluateConfig(parameters)
    applyConfig(parameters)
    print()
