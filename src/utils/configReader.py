import utils.config as config
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
        parameters[parts[0]] = parts[2]


def aplayConfig(parameters):
    config.NUMBER_OF_CLASSES = parameters["NUMBER_OF_CLASSES"]
    config.TRAINING_INFO = parameters["TRAINING_INFO"]
    config.TESTING_INFO = parameters["TESTING_INFO"]
    config.CLASSIFIERS = parameters["CLASSIFIERS"]
    config.TYPE_OF_DATA = parameters["TYPE_OF_DATA"]
    config.FEATURE_VECTOR = parameters["FEATURE_VECTOR"]
    if parameters["CUDA_USE"]:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        config.CUDA_USE = True
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        config.CUDA_USE = False

def readConfig():
    parameters = {}
    f = open(configFile, "r")
    for line in f.readlines():
        if line.startswith("#") or line.startswith(" ") or len(line) < 10:
            continue
        parserLine(line, parameters)
    f.close()
    aplayConfig(parameters)
