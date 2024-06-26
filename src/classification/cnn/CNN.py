import numpy as np
from classification.Classifier import Classifier, convert_labels_to_OneHot
from .CNN_model import create_CNN_model
from timeit import default_timer as timer
from ..Cross_Validation import reshape_data
from keras.callbacks import EarlyStopping
from utils.config import Config


class CNN(Classifier):

    def __init__(self):
        self.model = None
        self.name = "CNN"

    def train(self, data, labels) -> float:
        y_train = convert_labels_to_OneHot(labels)
        data = reshape_data(data)
        self.model = create_CNN_model(data[0].shape, np.array(y_train[0]).shape)
        callback = EarlyStopping(monitor='loss', patience=1)
        start = timer()
        self.model.fit(np.array(data), y_train, epochs=50, shuffle=True, verbose=Config.TRAINING_INFO, callbacks=[callback], batch_size=10)
        end = timer()
        time_of_training = end - start
        return time_of_training

    def validate(self, data, labels) -> [float, float]:
        y_train = convert_labels_to_OneHot(labels)
        data = reshape_data(data)
        scores = self.model.evaluate(np.array(data), y_train, verbose=Config.TESTING_INFO)
        start = timer()
        self.model.evaluate(np.array([data[0]]), np.array([y_train[0]]), verbose=0)
        end = timer()
        time_of_classification = end - start

        return scores[1] * 100, time_of_classification
