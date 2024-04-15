import tensorflow as tf
from keras.layers import Dense, BatchNormalization, Flatten
from keras import Sequential


def create_MLP_model(input_size, output_size):

    model = Sequential(name="MLP_model")
    initializer = tf.keras.initializers.GlorotUniform(seed=42)

    model.add(Dense(input_size[0], activation=tf.nn.relu, kernel_initializer=initializer))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(int(input_size[0]*2/3), activation=tf.nn.relu, kernel_initializer=initializer))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(int(input_size[0]/5), activation=tf.nn.relu, kernel_initializer=initializer))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(output_size[0], activation='softmax', kernel_initializer=initializer))

    # model.summary()
    #
    # noise = Input(shape=input_size)
    # img = model(noise)
    # return Model(noise, img)

    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=["accuracy"])

    return model


def create_MLP_model_Kodera(input_size, output_size):
    model = Sequential([
        Flatten(),
        Dense(512, activation='sigmoid'),
        Dense(256, activation='sigmoid'),
        Dense(128, activation='sigmoid'),
        Dense(output_size[0], activation='softmax')
    ], name="MLP_model_Kodera")

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
