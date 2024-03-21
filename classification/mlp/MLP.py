import numpy as np
from keras.optimizers import Adam
import keras
from classification.Classifier import Classifier
from .MLP_model import create_MLP_model, convert_labels_to_OneHot


class MLP(Classifier):

    def __init__(self):
        self.model = None
        self.optimizer = Adam(0.0002, 0.5)
        self.name = "MLP"

    def train(self, data, labels):
        y_train = convert_labels_to_OneHot(labels)
        self.model = create_MLP_model(data[0].shape, y_train[0].shape)
        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        # loss = self.model.train_on_batch(x=, y=np.array(labels))
        self.model.fit(np.array(data), y_train, epochs=60, shuffle=True, batch_size=25, verbose=2)
        # print(loss)

    def validate(self, data, labels) -> float:
        # output = self.model.predict(np.array(data))
        # print()
        y_train = convert_labels_to_OneHot(labels)
        scores = self.model.evaluate(np.array(data), y_train)
        print(self.model.metrics_names[0], self.model.metrics_names[1])

        return scores[1]
