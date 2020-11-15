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


def rosa_read(filename):
    y, sr = librosa.load(filename.numpy().decode("UTF-8"), sr=16000)
    y = librosa.util.fix_length(y, 16000)
    return y


def read_audio(filename, label):
    # read audio
    waveform = tf.py_function(features.audio_transform, [filename, label], [tf.float32])
    waveform = tf.convert_to_tensor(waveform)
    waveform = tf.squeeze(waveform, axis=0)
    # audio_binary = tf.io.read_file(filename)
    # waveform, _ = tf.audio.decode_wav(audio_binary)
    # waveform = tf.squeeze(waveform, axis=-1)

    # # padding if the audio less than 16k
    # zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)
    # waveform = tf.cast(waveform, tf.float32)
    # waveform = tf.concat([waveform, zero_padding], 0)

    return waveform, label # tf.one_hot(label, depth=8)

def main():
    """ ------------------- Features Configuration ------------------- """
    all_files = features.get_datafiles("training")

    # transform the list dicts into dataframe
    training_data = pd.DataFrame(all_files)
    training_data["label"] = [features.word_to_index[label] for label in training_data["label"]]
    # print(training_data.head(20))
    # print(min(training_data["label"]), max(training_data["label"]))
    training_data.to_csv("data/training_data.csv", index=False)

    # consume the training data to tf dataset
    training_dataset = tf.data.Dataset.from_tensor_slices(
        (training_data['file'], training_data['label'])
    )
    file, label = next(iter(training_dataset))
    print(read_audio(file, label))

    for feat, targ in training_dataset.take(5):
       print ('Features: {}, Target: {}'.format(feat, targ))





if __name__ == '__main__':
    main()
