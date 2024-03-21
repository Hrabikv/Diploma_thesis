import numpy as np
from keras.layers import Reshape, Input, Dense, BatchNormalization
from keras.models import Model
from keras import Sequential
from keras.layers.activation import LeakyReLU

di = {
    2: [1, 0, 0],
    5: [0, 1, 0],
    6: [0, 0, 1]
    }


def convert_labels_to_OneHot(labels):
    result = []
    for label in labels:
        result.append(np.array(di[label]))
    return np.array(result)


def create_MLP_model(input_size, output_size):

    model = Sequential(name="MLP_model")

    model.add(Dense(2500, input_shape=input_size))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1500))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(250))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(output_size), activation='tanh'))
    model.add(Dense(3, activation='softmax'))

    # model.summary()

    noise = Input(shape=input_size)
    img = model(noise)
    return Model(noise, img)
