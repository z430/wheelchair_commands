import random
import time

import keras
import numpy as np
import tensorflow as tf
from keras import losses, optimizers
from keras.backend import set_session
from keras.callbacks import TensorBoard

import dnn_models
import input_data

r = random.randint(1111, 9999)


def data_gen(sess, features_settings, mode='training', batch_size=5):
    offset = 0
    if mode != 'training':
        background_frequency = 0.0
        background_volume_range = 0.0
        foreground_frequency = 0.0
        foreground_volume_range = 0.0
        pseudo_frequency = 0.0
        time_shift_frequency = 0.0
        time_shift_range = [0, 0]
    while True:
        X, y = features_settings.get_data(
            how_many=batch_size, offset=0 if mode == 'training' else offset,
            mode=mode)

        offset += batch_size
        if offset > features_settings.set_size(mode) - batch_size:
            offset = 0
        yield X, y


def main():
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.7
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    """ ------------------- GET TF SESSION ------------------- """
    sess = tf.InteractiveSession()
    use_gen = False

    """ ------------------- Features Configuration ------------------- """
    wanted_words = 'left,right,forward,backward,stop,go'
    speech_feature = 'mfcc'
    features = input_data.GetData(
        wanted_words=wanted_words, feature=speech_feature)
    # initialize dataset
    # features.initialize()
    model_settings = features.model_settings

    sr = random.SystemRandom()
    # version number is random every training
    version_number = (time.asctime(
        time.localtime(time.time()))).replace(' ', '_')
    """ ------------------- Model Configuration ------------------- """
    max_epoch = 50
    learning_rate = 0.001
    decay_rate = learning_rate / max_epoch
    opt = optimizers.RMSprop(lr=learning_rate)
    model_name = 'cnn_trad_fpool3'

    input_size = np.ones(features.sample_rate)
    input_size = features.speech_features.mfcc(input_size)
    input_size_flat = input_size.flatten()
    dnn_model = dnn_models.select_model(input_size_flat.shape, model_settings['label_count'],
                                        model_name)
    dnn_model.compile(
        optimizer=opt, loss=losses.categorical_crossentropy,
        metrics=[
            'accuracy',
            'categorical_accuracy'
        ]
    )
    dnn_model.summary()

    feature_name = f"{speech_feature}_{input_size.shape[0]}x{input_size.shape[1]}_{wanted_words.replace(',', '_')}"
    if use_gen:
        train_gen = data_gen(sess, features, mode='training')
        val_gen = data_gen(sess, features, mode='validation')
    else:
        x_train = np.load(f"data/x_train_{feature_name}.npy")
        y_train = np.load(f"data/y_train_{feature_name}.npy")

        x_val = np.load(f"data/x_val_{feature_name}.npy")
        y_val = np.load(f"data/y_val_{feature_name}.npy")
    feature_name = f"{model_name}_{speech_feature}_{input_size.shape[0]}x{input_size.shape[1]}_" \
                   f"{wanted_words.replace(',', '_')}"
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath='models/{}.hdf5'.format(feature_name),
        verbose=1, save_best_only=True,
        monitor='val_categorical_accuracy')
    model_stop_training = keras.callbacks.EarlyStopping(
        monitor='loss', patience=100, verbose=1)
    tensorboard = TensorBoard(log_dir='./retrain_logs', histogram_freq=0)
    lr_reduce_op = keras.callbacks.ReduceLROnPlateau(
        factor=0.01, min_lr=0.001, monitor='categorical_accuracy')

    batch_size = 32

    """ -------------------------------------- train ----------------------------------------------"""
    print(50 * '=', 'STAGE 2 TRAINING', 50 * '=')
    if use_gen:
        dnn_model.fit_generator(train_gen,
                                steps_per_epoch=features.set_size(
                                    'training') // batch_size,
                                epochs=100, verbose=1, callbacks=[
                tensorboard,
                model_checkpoint,
                lr_reduce_op, model_stop_training],
                                validation_data=val_gen, validation_steps=features.set_size('validation') // batch_size)
    else:
        dnn_model.fit(
            x=x_train, y=y_train, batch_size=batch_size, epochs=max_epoch,
            callbacks=[
                tensorboard, model_checkpoint, lr_reduce_op, model_stop_training
            ],
            validation_data=(x_val, y_val)
        )


if __name__ == '__main__':
    main()
