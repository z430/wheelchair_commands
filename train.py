""" Training """
import os
import random
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models

import pandas as pd
from audio_utility import AudioUtil
import librosa

import input_data

au = AudioUtil(desired_samples=16000, sample_rate=16000, normalize=False)
wanted_words = 'left,right,forward,backward,stop,go'
features = input_data.GetData(wanted_words=wanted_words, feature="mfcc")
AUTOTUNE = tf.data.experimental.AUTOTUNE

def rosa_read(filename, label):
    waveform = tf.py_function(features.audio_transform, [filename, label], [tf.float32])
    waveform = tf.convert_to_tensor(waveform)
    waveform = tf.squeeze(waveform, axis=0)
    return waveform, label

def get_spectrogram(waveform, label):
    waveform = tf.cast(waveform, tf.float32)
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=-1)
    return spectrogram, label

def preprocess_dataset(dataset):
    files_ds = tf.data.Dataset.from_tensor_slices(
        (dataset['file'], dataset['label'])
    )
    output_ds = files_ds.map(rosa_read, num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(get_spectrogram, num_parallel_calls=AUTOTUNE)
    return output_ds


def main():
    """ ------------------- Features Configuration ------------------- """
    training_files = features.get_datafiles('training')
    validation_files = features.get_datafiles('validation')

    # transform the list dicts into dataframe
    training_data = pd.DataFrame(training_files)
    training_data['label'] = [features.word_to_index[label] for label in training_data['label']]

    validation_data = pd.DataFrame(validation_files)
    validation_data['label'] = [features.word_to_index[label] for label in validation_data['label']]

    training_ds = preprocess_dataset(training_data)
    validation_ds = preprocess_dataset(validation_data)

    batch_size = 32
    training_ds = training_ds.batch(batch_size)
    validation_ds = validation_ds.batch(batch_size)

    training_ds = training_ds.cache().prefetch(AUTOTUNE)
    validation_ds = validation_ds.cache().prefetch(AUTOTUNE)

    for spectrogram, labels in training_ds.take(1):
        input_shape = spectrogram.shape[1:]
        print(input_shape, labels)

    num_labels = len(features.words_list)
    print(f"Input Shape: {input_shape}, len labels: {num_labels}")

    norm_layer = preprocessing.Normalization()
    norm_layer.adapt(training_ds.map(lambda x, _: x))

    model = models.Sequential([
        layers.Input(shape=input_shape),
        preprocessing.Resizing(32, 32),
        norm_layer,
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels),
    ])

    # callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(verbose=1, patience=2)

    checkpoint_path = "checkpoints/spectrogram/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoints = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, save_weights_only=True, verbose=1
    )

    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    EPOCHS = 100
    history = model.fit(
        training_ds,
        validation_data=validation_ds,
        epochs=EPOCHS,
        callbacks=[early_stopping, checkpoints],
    )

if __name__ == '__main__':
    main()
