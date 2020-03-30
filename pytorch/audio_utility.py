import python_speech_features as psf
import os
import numpy as np
from pycochleagram import cochleagram as cgram
import librosa


class SpeechFeatures:

    def __init__(self):
        self.fs = 16000
        self.n_fft = 512
        self.win_length = int(self.fs * (30 / 1000))  # 40ms
        self.hop_length = int(self.fs * (10 / 1000))  # 40ms
        print(f"win length: {self.win_length} hop length: {self.hop_length}")

    # Mel Frequency Cepstral Coefficient (MFCC)
    def mfcc(self, sig):
        num_filters = 10
        mfcc = psf.mfcc(sig, winlen=0.030, winstep=0.01,
                        nfft=512, nfilt=num_filters,
                        numcep=40)
        return mfcc

    def cgram_(self, y, fs):
        cg = cgram.human_cochleagram(
            y, fs, n=38, sample_factor=2, downsample=10, nonlinearity='power', strict=False)
        return cg


class AudioUtil:

    def __init__(self, model_settings):
        self.model_settings = model_settings

    def fix_audio_length(self, audio):
        desired_samples = self.model_settings['desired_samples']
        return librosa.util.fix_length(audio, desired_samples)

    def processing_audio(self, input_data, normalize=False):
        # read audio
        y, sr = librosa.load(input_data['wav_filename'],
                             sr=self.model_settings['sample_rate'])
        y = self.fix_audio_length(y)
        # pre emphasis
        y = psf.sigproc.preemphasis(y)
        scaled_foreground = y * input_data['foreground_volume']
        padded_foreground = np.pad(
            scaled_foreground,
            input_data['time_shift_padding'],
            'constant'
        )

        time_shift_offset = input_data['time_shift_offset']
        sliced_foreground = self.fix_audio_length(
            padded_foreground[time_shift_offset:])
        background_mul = np.multiply(
            input_data['background_data'], input_data['background_volume'])
        audio_result = np.add(background_mul, sliced_foreground)
        if normalize:
            audio_result = np.clip(audio_result, -1.0, 1.0)
        return audio_result


if __name__ == '__main__':
    feature = SpeechFeatures()
    sig = np.ones(16000)
    mfcc = feature.mfcc(sig)
    print(mfcc.shape)
