import keras
import librosa
import numpy as np

import audio_utility as au

feature_extractor = au.SpeechFeatures()
wanted_words = ['_silence_', '_unknown_', 'backward', 'forward', 'go', 'left', 'right', 'stop']

# import keras model
model = keras.models.load_model('models/vgg_19_model_mfcc_49x40_left_right_forward_backward_stop_go.hdf5')
model.summary()

# load test audio
test_audio = 'data/train/right/0b09edd3_nohash_0.wav'
# read test audio
y, sr = librosa.load(test_audio, sr=16000)
# fix the length of test audio to 1s or 16000 samples
y = librosa.util.fix_length(y, size=16000)
# normalize the test audio into [-1. 1]
# y = np.clip(y, -1.0, 1.0)

# extract the features
audio_feature = feature_extractor.mfcc(y, fs=16000)
print(audio_feature.shape)
audio_feature = audio_feature.flatten()
audio_feature = np.expand_dims(audio_feature, axis=0)
# print(audio_feature.shape)

# # inference
prediction = model.predict(audio_feature)
print(prediction)

prediction = np.argmax(prediction)
print(prediction)
print(wanted_words[int(prediction)])
print(wanted_words)
