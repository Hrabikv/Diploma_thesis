from ..Classifier import Classifier, convert_labels_to_OneHot
import numpy as np
from timeit import default_timer as timer
from keras.callbacks import EarlyStopping
from .Transformer_model import create_Transformer_model
from src.config import TRAINING_INFO, TESTING_INFO


class Transformer(Classifier):

    def __init__(self):
        self.model = None
        self.name = "Transformer"

    def train(self, data, labels) -> float:
        y_train = convert_labels_to_OneHot(np.array(labels))
        x_train = np.array(data)
        x_train = x_train.reshape((x_train.shape[0], 3, int(x_train.shape[1]/3)))
        idx = np.random.permutation(len(x_train))
        x_train = x_train[idx]
        y_train = y_train[idx]
        y_train[y_train == -1] = 0
        input_shape = x_train.shape[1:]
        self.model = create_Transformer_model(
            input_shape,
            head_size=250,
            num_heads=4,
            ff_dim=4,
            num_transformer_blocks=8,
            mlp_units=[125],
            mlp_dropout=0.4,
            dropout=0.25,
        )
        callback = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        start = timer()
        self.model.fit(x_train, y_train, epochs=50, shuffle=True, verbose=TRAINING_INFO, callbacks=[callback], batch_size=10)
        end = timer()
        time_of_training = end - start
        return time_of_training

    def validate(self, data, labels) -> [float, float]:
        y_train = convert_labels_to_OneHot(np.array(labels))
        y_train[y_train == -1] = 0
        x_test = np.array(data)
        x_test = x_test.reshape((x_test.shape[0], 3, int(x_test.shape[1]/3)))
        scores = self.model.evaluate(x_test, y_train, verbose=TESTING_INFO)
        start = timer()
        self.model.evaluate(np.array([x_test[0]]), np.array([y_train[0]]), verbose=0)
        end = timer()
        time_of_classification = end - start

        return scores[1] * 100, time_of_classification