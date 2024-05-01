import tensorflow as tf
from keras.layers import Dense, BatchNormalization, LSTM
from keras import Sequential
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def create_LSTM_model(input_size, output_size):

    model = Sequential(name="LSTM_model")
    initializer = tf.keras.initializers.GlorotUniform(seed=int(42))

    model.add(LSTM(250, activation=tf.nn.relu, return_sequences=True, kernel_initializer=initializer,
                   input_shape=input_size))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LSTM(150, activation=tf.nn.relu, kernel_initializer=initializer))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(output_size[0], activation='softmax', kernel_initializer=initializer))

    # model.summary()

    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=["accuracy"])

    return model

def create_LSTM_model_Kodera(input_size, output_size):
    dropout = 0.2

    layers = [
        LSTM(256, return_sequences=True, dropout=dropout),
        LSTM(256, return_sequences=True, dropout=dropout),
        LSTM(256, return_sequences=False, dropout=dropout),
        Dense(output_size[0], activation='softmax')
    ]

    model = Sequential(layers, name="LSTM_Kodera")

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model