import python_speech_features as psf
import numpy as np
# from pycochleagram import cochleagram as cgram
import librosa


class SpeechFeatures:

    def __init__(self):
        self.fs = 16000

        # mfcc 49x40
        self.win_length = 0.04
        self.hop_length = 0.02
        self.numcep = 40
        self.nfilt = 40
        self.n_fft = 1024

    # Mel Frequency Cepstral Coefficient (MFCC)
    def mfcc(self, sig, fs):
        # print(self.win_length, self.hop_length)
        mfcc = psf.mfcc(
            sig, winlen=self.win_length, nfft=self.n_fft,
            winstep=self.hop_length, numcep=self.numcep, nfilt=self.nfilt
        )
        # print(mfcc.shape)
        return mfcc

    # def cgram_(self, y, fs):
    #     cg = cgram.human_cochleagram(
    #         y, fs, n=20, sample_factor=2, downsample=20, nonlinearity='power', strict=False)
    #     return cg


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

