import tensorflow as tf
from keras.models import Model
import keras.layers as kl
from keras import Sequential
from src.config import FEATURE_VECTOR


def create_CNN_model(input_shape, output_shape):
    initializer = tf.keras.initializers.GlorotUniform(seed=42)

    input_layer = kl.Input(input_shape)

    if FEATURE_VECTOR == "time":
        filter_size = 100
        kernel_size = 5
    elif FEATURE_VECTOR == "freq":
        filter_size = 10
        kernel_size = 5

    conv1 = kl.Conv1D(filters=filter_size, kernel_size=kernel_size, padding="same", kernel_initializer=initializer)(input_layer)
    conv1 = kl.BatchNormalization()(conv1)
    conv1 = kl.ReLU()(conv1)

    conv2 = kl.Conv1D(filters=filter_size, kernel_size=kernel_size, padding="same", kernel_initializer=initializer)(conv1)
    conv2 = kl.BatchNormalization()(conv2)
    conv2 = kl.ReLU()(conv2)

    conv3 = kl.Conv1D(filters=filter_size, kernel_size=kernel_size, padding="same", kernel_initializer=initializer)(conv2)
    conv3 = kl.BatchNormalization()(conv3)
    conv3 = kl.ReLU()(conv3)

    gap = kl.GlobalAveragePooling1D()(conv3)

    output_layer = kl.Dense(output_shape[0], activation="softmax")(gap)

    model = Model(inputs=input_layer, outputs=output_layer, name="CNN_model")
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
