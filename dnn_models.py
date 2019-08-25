import keras
import keras.backend as K
from keras.layers import *
from keras import models
from keras.regularizers import l2
from keras.applications import InceptionV3
from keras.activations import softmax
from keras.models import Sequential
from keras.constraints import maxnorm


def create_cnn_models(input_shape, nclass):
    model = Sequential()
    # model.add(Reshape([98, 10, 1], input_shape=input_shape))  # mfcc
    model.add(Conv2D(filters=60, kernel_size=(10, 4), strides=(1, 1), activation='relu', name='conv1',
                     input_shape=(81, 10, 1)))
    model.add(Conv2D(filters=30, kernel_size=(10, 4), strides=(2, 1), activation='relu', name='conv2', ))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(nclass, activation='softmax', name='fp2'))
    return model


def create_dnn_models(input_shape, nclass):
    model = Sequential()
    model.add(Input(input_shape))


def select_model(input_shape, nclass, model_name):
    if model_name == 'cnn':
        return create_cnn_models(input_shape, nclass)
