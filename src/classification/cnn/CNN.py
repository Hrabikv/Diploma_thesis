import numpy as np
from src.classification.Classifier import Classifier, convert_labels_to_OneHot
from .CNN_model import create_CNN_model
from timeit import default_timer as timer
from ..Cross_Validation import reshape_data
from keras.callbacks import EarlyStopping


class CNN(Classifier):

    def __init__(self):
        self.model = None
        self.name = "CNN"

    def train(self, data, labels) -> float:
        y_train = convert_labels_to_OneHot(labels)
        # length = len(data[0])
        # step = int(length/3)
        # sample = np.array(data[0])
        #
        # data_shape = np.array([sample[:step], sample[step:step*2], sample[step*2:]])
        data = reshape_data(data)
        self.model = create_CNN_model(data[0].shape, np.array(y_train[0]).shape)
        callback = EarlyStopping(monitor='loss', patience=3)
        start = timer()
        self.model.fit(np.array(data), y_train, epochs=50, shuffle=True, verbose=0, callbacks=[callback])
        end = timer()
        time_of_training = end - start
        # plot_history_of_MLP(history, self.model.metrics_names[1])
        return time_of_training

    def validate(self, data, labels) -> [float, float]:
        y_train = convert_labels_to_OneHot(labels)
        data = reshape_data(data)
        scores = self.model.evaluate(np.array(data), y_train)
        start = timer()
        self.model.evaluate(np.array([data[0]]), np.array([y_train[0]]))
        end = timer()
        time_of_classification = end - start

        return scores[1] * 100, time_of_classification
