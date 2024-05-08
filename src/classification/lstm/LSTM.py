from classification.Classifier import Classifier, convert_labels_to_OneHot
from .LSTM_model import create_LSTM_model
from timeit import default_timer as timer
from keras.callbacks import EarlyStopping
import numpy as np
from utils.config import Config


class LSTM(Classifier):

    def __init__(self):
        self.model = None
        self.name = "LSTM"

    def train(self, data, labels) -> float:
        y_train = convert_labels_to_OneHot(labels)
        data = np.reshape(data, (len(data), 1, len(data[0])))
        self.model = create_LSTM_model(np.array(data[0]).shape, y_train[0].shape)
        callback = EarlyStopping(monitor='loss', patience=1, min_delta=0.01)
        start = timer()
        self.model.fit(np.array(data), y_train, epochs=50, shuffle=True, verbose=Config.TRAINING_INFO, callbacks=[callback], batch_size=10)
        end = timer()
        time_of_training = end - start
        return time_of_training

    def validate(self, data, labels) -> [float, float]:
        y_train = convert_labels_to_OneHot(labels)
        data = np.reshape(data, (len(data), 1, len(data[0])))
        scores = self.model.evaluate(np.array(data), y_train, verbose=Config.TESTING_INFO)
        start = timer()
        self.model.evaluate(np.array([data[0]]), np.array([y_train[0]]), verbose=0)
        end = timer()
        time_of_classification = end - start

        return scores[1] * 100, time_of_classification
