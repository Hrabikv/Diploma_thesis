import tensorflow as tf
from keras.models import Model
import keras.layers as kl
from keras import Sequential
from utils.config import Config
from keras.constraints import max_norm


def create_CNN_model(input_shape, output_shape):
    initializer = tf.keras.initializers.GlorotUniform(seed=42)

    input_layer = kl.Input(input_shape)

    if Config.FEATURE_VECTOR == "time":
        filter_size = 100
        kernel_size = 5
    elif Config.FEATURE_VECTOR == "freq":
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


def create_CNN_model_Kodera(input_shape, output_shape):
    height = input_shape[0]
    conv_depth = 2
    filters_block1 = 8
    filters_block2 = 16
    kernel_size_block1 = 64
    kernel_size_block2 = 16
    dropout_rate = 0.5
    pool_size = 4
    model = Sequential([
        # Block 1
        kl.Conv1D(filters_block1, kernel_size_block1, padding='same', input_shape=input_shape),
        kl.BatchNormalization(),
        kl.DepthwiseConv1D(kernel_size_block1, depth_multiplier=conv_depth, depthwise_constraint=max_norm(1.)),
        kl.BatchNormalization(),
        kl.Activation('relu'),
        kl.AveragePooling1D(pool_size=pool_size),
        kl.Dropout(dropout_rate),

        # Block 2
        kl.SeparableConv1D(filters_block2, kernel_size_block2, padding='same'),
        kl.BatchNormalization(),
        kl.Activation('relu'),
        kl.AveragePooling1D(pool_size=pool_size),
        kl.Dropout(dropout_rate),

        kl.Flatten(),

        kl.Dense(output_shape[0], activation='softmax')
    ], name="CNN_Kodera")


    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
