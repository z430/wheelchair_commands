import keras
import librosa
from keras import optimizers, losses
from keras.backend import set_session
from keras.callbacks import TensorBoard

import tensorflow as tf
import pandas as pd
import h5py

from sklearn.metrics import roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
import argparse
import input_data
import numpy as np
import dnn_models
import random
import audio_utility as au
import sys
import time
import os

r = random.randint(1111, 9999)


def data_gen(sess, features_settings, mode='training', batch_size=-1):
    offset = 0
    if mode != 'training':
        background_frequency = 0.0
        background_volume_range = 0.0
        foreground_frequency = 0.0
        foreground_volume_range = 0.0
        pseudo_frequency = 0.0
        time_shift_frequency = 0.0
        time_shift_range = [0, 0]
    # while True:
    x, y = features_settings.get_data(
        how_many=batch_size, offset=0 if mode == 'training' else offset,
        mode=mode)
    if mode is "training":
        np.save("features/x_train_cgram.npy", x)
        np.save("features/y_train_cgram.npy", y)
    elif mode is "validation":
        np.save("features/x_validation_cgram.npy", x)
        np.save("features/y_validation_cgram.npy", y)
    elif mode is "testing":
        np.save("features/x_testing_cgram.npy", x)
        np.save("features/y_testing_cgram.npy", y)
    # offset += batch_size
    # if offset > features_settings.set_size(mode) - batch_size:
    #     offset = 0

    # yield x, y


def main():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    """ ------------------- GET TF SESSION ------------------- """
    sess = tf.InteractiveSession()

    """ ------------------- Features Configuration ------------------- """
    wanted_words = 'left,right,forward,backward,stop,go'
    speech_feature = 'cgram'
    features = input_data.GetData(wanted_words=wanted_words, feature=speech_feature)
    # initialize dataset
    features.initialize()
    model_settings = features.model_settings

    sr = random.SystemRandom()
    # version number is random every training
    version_number = (time.asctime(time.localtime(time.time()))).replace(' ', '_')
    """ ------------------- Model Configuration ------------------- """

    if not os.path.exists(f"features/x_train_{speech_feature}.npy"):
        data_gen(sess, features, mode='training')
    # sys.exit()
    if not os.path.exists(f"features/x_validation_{speech_feature}.npy"):
        data_gen(sess, features, mode='validation')
    if not os.path.exists(f"features/x_testing_{speech_feature}.npy"):
        data_gen(sess, features, mode='testing')

    xtrain = np.load(f"features/x_train_{speech_feature}.npy")
    ytrain = np.load(f"features/y_train_{speech_feature}.npy")

    xval = np.load(f"features/x_validation_{speech_feature}.npy")
    yval = np.load(f"features/y_validation_{speech_feature}.npy")

    print("silence percentage: ", features.silence_percentage,
          "unknown percentage: ", features.unknown_percentage)

    max_epoch = 1000
    learning_rate = 0.0001
    decay_rate = learning_rate / max_epoch
    opt = optimizers.RMSprop(lr=learning_rate)
    model_name = 'cnn'

    input_size = features.input_shape
    print(f"input shape: {input_size}")
    dnn_model = dnn_models.select_model(input_size, model_settings['label_count'],
                                        model_name)
    dnn_model.compile(
        optimizer=opt, loss=losses.categorical_crossentropy,
        metrics=[
            'accuracy',
            'categorical_accuracy'
        ]
    )

    model_name = f"{version_number}_{speech_feature}_{wanted_words.replace(',', '_')}"
    dnn_model.summary()
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath='models/{}_{}.hdf5'.format(model_name, r),
        verbose=1, save_best_only=False,
        monitor='val_categorical_accuracy')
    model_stop_training = keras.callbacks.EarlyStopping(
        monitor='loss', patience=100, verbose=1)
    tensorboard = TensorBoard(log_dir=f"retrain_logs/{version_number}", histogram_freq=0)
    lr_reduce_op = keras.callbacks.ReduceLROnPlateau(
        factor=0.01, min_lr=0.00001, monitor='categorical_accuracy')
    batch_size = 10

    """ -------------------------------------- train ----------------------------------------------"""
    print(50 * '=', 'STAGE 2 TRAINING', 50 * '=')
    # dnn_model.fit_generator(train_gen,
    #                         steps_per_epoch=features.set_size(
    #                             'training') // batch_size,
    #                         epochs=100, verbose=1, callbacks=[
    #         tensorboard,
    #         model_checkpoint,
    #         lr_reduce_op, model_stop_training])
    xtrain = np.reshape(xtrain, (xtrain.shape[0], 81, 10, 1))
    xval = np.reshape(xval, (xval.shape[0], 81, 10, 1))
    dnn_model.fit(xtrain, ytrain, batch_size=10, epochs=max_epoch, validation_data=(xval, yval),
                  callbacks=[
                      tensorboard,
                      model_checkpoint,
                      lr_reduce_op, model_stop_training],
                  )

    dnn_model.save(f"models/{model_name}.hdf5")


if __name__ == '__main__':
    main()
