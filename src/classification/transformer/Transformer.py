from ..Classifier import Classifier, convert_labels_to_OneHot
import numpy as np
from timeit import default_timer as timer
from keras.callbacks import EarlyStopping
from .Transformer_model import create_Transformer_model


class Transformer(Classifier):

    def __init__(self):
        self.model = None

    def train(self, data, labels) -> float:
        y_train = convert_labels_to_OneHot(np.array(labels))
        x_train = np.array(data)
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
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
            num_transformer_blocks=4,
            mlp_units=[125],
            mlp_dropout=0.4,
            dropout=0.25,
        )
        callback = EarlyStopping(monitor='loss', patience=1, restore_best_weights=True)
        start = timer()
        self.model.fit(x_train, y_train, epochs=150, shuffle=True, verbose=1, callbacks=[callback], batch_size=5)
        end = timer()
        time_of_training = end - start
        return time_of_training

    def validate(self, data, labels) -> [float, float]:
        y_train = convert_labels_to_OneHot(np.array(labels))
        y_train[y_train == -1] = 0
        x_test = np.array(data)
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
        scores = self.model.evaluate(x_test, y_train, verbose=0)
        start = timer()
        self.model.evaluate([x_test[0]], np.array([y_train[0]]), verbose=0)
        end = timer()
        time_of_classification = end - start

        return scores[1] * 100, time_of_classification