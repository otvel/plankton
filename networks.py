from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.core import Activation, Dense, Flatten, Dropout
from keras.layers.normalization import BatchNormalization


class SimpleFF:
    @staticmethod
    def build(input_shape, classes):
        """"A simple feed forward model"""
        model = Sequential()
        model.add(Dense(512, input_shape=input_shape, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.75))
        model.add(Dense(classes, activation='softmax'))

        return model


class MyLeNet:
    @staticmethod
    def build(rows, cols, chans, classes):
        """My variation of LeNet"""

        input_shape = (rows, cols, chans)
        model = Sequential()
        # 6 filters, 5x5 kernel
        model.add(Conv2D(6, 5, strides=(1, 1), padding="valid", 
                         input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))
        # 16 filters, 5x5 kernel 
        model.add(Conv2D(16, 5, strides=(1, 1), padding="valid"))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))
        # FC
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model


class MyCNN:
    @staticmethod
    def build(rows, cols, chans, classes):
        """My variation of a slightly deeper CNN"""

        input_shape = (rows, cols, chans)
        model = Sequential()
        # Conv
        model.add(Conv2D(32, 3, strides=(2, 2), padding='same',
                         input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Conv2D(32, 3, strides=(1, 1), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(64, 3, strides=(1, 1), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, 3, strides=(1, 1), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(96, 3, strides=(1, 1), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(128, 3, strides=(1, 1), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # FC
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(classes, activation='softmax'))

        return model


class LeNet:
    @staticmethod
    def build(rows, cols, chans, classes):
        """Original LeNet"""

        input_shape = (rows, cols, chans)
        model = Sequential()
        # 6 filters, 5x5 kernel
        model.add(Conv2D(6, 5, strides=(1, 1), padding="valid",
                         input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
        # 16 filters, 5x5 kernel 
        model.add(Conv2D(16, 5, padding="valid"))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
        # FC
        model.add(Flatten())
        model.add(Dense(120))
        model.add(Activation('relu'))
        model.add(Dense(84))
        model.add(Activation('relu'))
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model


def save_model_structure(model, output_path):
    with open(output_path, 'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
