import numpy as np
import tensorflow as tf
from keras.layers import Reshape, Input, Dense, BatchNormalization, LSTM
from keras.models import Model
from keras import Sequential
from keras.metrics import Accuracy, CategoricalAccuracy, SparseCategoricalAccuracy
from keras.layers.activation import LeakyReLU
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def create_LSTM_model(input_size, output_size):

    model = Sequential(name="LSTM_model")
    initializer = tf.keras.initializers.GlorotUniform(seed=int(42))

    model.add(LSTM(250, activation=tf.nn.relu, return_sequences=True, kernel_initializer=initializer,
                   input_shape=input_size))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LSTM(150, activation=tf.nn.relu, kernel_initializer=initializer))
    # model.add(BatchNormalization(momentum=0.8))
    # model.add(LSTM(250, activation=tf.nn.relu, kernel_initializer=initializer))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(output_size[0], activation='softmax', kernel_initializer=initializer))

    # model.summary()
    #
    # noise = Input(shape=input_size)
    # img = model(noise)
    # return Model(noise, img)

    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=["accuracy"])

    return model