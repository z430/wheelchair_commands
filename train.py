""" Training """

import random
import time

import keras
import numpy as np
import tensorflow as tf

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

    training_ds = training_ds.cache().prefetch(AUTOTUNE)
    validation_ds = validation_ds.cache().prefetch(AUTOTUNE)


if __name__ == '__main__':
    main()
