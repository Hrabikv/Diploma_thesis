import tensorflow as tf
from keras.layers import Reshape, Input, Dense, BatchNormalization, Conv1D, GlobalAveragePooling1D
from keras.models import Model
import keras.layers as kl
from keras import Sequential
import numpy as np


def create_CNN_model(input_shape, output_shape):
    model = Sequential(name="CNN_model")
    initializer = tf.keras.initializers.GlorotUniform(seed=42)

    input_layer = kl.Input(input_shape)

    conv1 = kl.Conv1D(filters=100, kernel_size=5, padding="same", kernel_initializer=initializer)(input_layer)
    conv1 = kl.BatchNormalization()(conv1)
    conv1 = kl.ReLU()(conv1)

    conv2 = kl.Conv1D(filters=100, kernel_size=5, padding="same", kernel_initializer=initializer)(conv1)
    conv2 = kl.BatchNormalization()(conv2)
    conv2 = kl.ReLU()(conv2)

    conv3 = kl.Conv1D(filters=100, kernel_size=5, padding="same", kernel_initializer=initializer)(conv2)
    conv3 = kl.BatchNormalization()(conv3)
    conv3 = kl.ReLU()(conv3)

    gap = kl.GlobalAveragePooling1D()(conv3)

    output_layer = kl.Dense(output_shape[0], activation="softmax")(gap)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=["accuracy"])

    return model

    # input_layer = Input(input_shape)
    #
    # model.add(input_layer)
    # model.add(Conv1D(filters=64, kernel_size=3, padding="same", activation=tf.nn.relu, kernel_initializer=initializer))
    # model.add(BatchNormalization(momentum=0.8))
    # model.add(Conv1D(filters=64, kernel_size=3, padding="same",  activation=tf.nn.relu, kernel_initializer=initializer))
    # model.add(BatchNormalization(momentum=0.8))
    # model.add(Conv1D(filters=64, kernel_size=3, padding="same",  activation=tf.nn.relu, kernel_initializer=initializer))
    # model.add(BatchNormalization(momentum=0.8))
    # model.add(GlobalAveragePooling1D(output_shape[0], activation='softmax', kernel_initializer=initializer))
    #
    # model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=["accuracy"])
    #
    # return model
