import hashlib
import math
import os.path
import random
import re
import numpy as np
import os
import re
import hashlib
import random
import math
import sys
import tarfile

import glob
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import tqdm

from keras import utils
import tensorflow as tf
import urllib

import audio_utility as audio_util


class GetData:

    def __init__(self, prepare_data=True, wanted_words='marvin', feature='cgram'):

        # don't change this parameter
        self.prepare_data = prepare_data
        self.MAX_NUM_WAVS_PER_CLASS = 2 ** 27 - 1  # ~134MB
        self.RANDOM_SEED = 59185
        self.SILENCE_INDEX = 0
        self.SILENCE_LABEL = '_silence_'
        self.UNKNOWN_WORD_INDEX = 1
        self.UNKNOWN_WORD_LABEL = '_unknown_'
        self.BACKGROUND_NOISE_DIR_NAME = '_background_noise_'

        self.data_url = 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
        self.data_dir = 'data/train'
        if os.path.isdir(self.data_dir):
            self.maybe_download_and_extract_dataset(self.data_url, self.data_dir)
        else:
            os.makedirs(self.data_dir)
            self.maybe_download_and_extract_dataset(self.data_url, self.data_dir)
        self.background_volume = 0.1
        self.background_frequency = 0.3
        self.time_shift_ms = 50.0
        self.sample_rate = 16000
        self.clip_duration_ms = 1000 / 1000  # ms to s
        self.window_size = 32.0
        self.window_stride = 10.0
        self.dct_coefficient_count = 40

        self.silence_percentage = 30.0
        self.unknown_percentage = 30.0
        self.testing_percentage = 10.0
        self.validation_percentage = 10.0
        self.how_many_training_steps = 15000
        self.learning_rate = 0.001
        self.batch_size = 100
        self.summaries_dir = 'retrain_logs'
        self.wanted_words = sorted(wanted_words.split(','))
        self.train_dir = 'wuw_train_logs'

        self.model_architecture = 'conv'
        self.audio_feature = 'mfcc'
        self.speech_features = audio_util.SpeechFeatures()

        # initialization
        self.model_settings = self.prepare_model_setting()
        self.audio_util = audio_util.AudioUtil(self.model_settings)
        self.words_list = self.prepare_word_list(self.wanted_words)
        # get input shape
        self.input_shape = self.get_input_shape(feature)

    def get_input_shape(self, feature_name):
        audio = np.ones(self.sample_rate * int(self.clip_duration_ms))
        print(f"audio length: ")
        if feature_name is "cgram":
            f = self.speech_features.cgram_(audio, self.sample_rate)
            return f.shape
        elif feature_name is "mfcc":
            f = self.speech_features.mfcc(audio)
            return f.shape
        else:
            return None

    def initialize(self):
        if self.prepare_data:
            self.prepare_data_index(self.silence_percentage, self.unknown_percentage, self.wanted_words,
                                    self.validation_percentage, self.testing_percentage)
            self.prepare_background_data()

    @staticmethod
    def prepare_word_list(wanted_words):
        return ['_silence_', '_unknown_'] + wanted_words

    def prepare_model_setting(self):
        desired_samples = int(self.sample_rate * self.clip_duration_ms)

        return {
            'desired_samples': desired_samples,
            'window_size_ms': self.window_size / 1000,
            'window_stride_ms': self.window_stride / 1000,
            'dct_coefficient_count': self.dct_coefficient_count,
            'audio_feature': self.audio_feature,
            'label_count': len(self.prepare_word_list(self.wanted_words)),
            'sample_rate': self.sample_rate
        }

    def maybe_download_and_extract_dataset(self, data_url, dest_directory):
        """
            Download and extract data set tar file.
            If the data set we're using doesn't already exist, this function
            downloads it from the TensorFlow.org website and unpacks it into a
            directory.
            If the data_url is none, don't download anything and expect the data
            directory to contain the correct files already.
            Args:
            data_url: Web location of the tar file containing the data set.
            dest_directory: File path to extract data to.
        """
        if not data_url:
            return
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)
        filename = data_url.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)
        if not os.path.exists(filepath):

            def _progress(count, block_size, total_size):
                sys.stdout.write(
                    '\r>> Downloading %s %.1f%%' %
                    (filename, float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()

            try:
                filepath, _ = urllib.request.urlretrieve(
                    data_url, filepath, _progress)
            except:
                tf.logging.error('Failed to download URL: %s to folder: %s', data_url,
                                 filepath)
                tf.logging.error('Please make sure you have enough free space and'
                                 ' an internet connection')
                raise
            print()
            statinfo = os.stat(filepath)
            tf.logging.info('Successfully downloaded %s (%d bytes)', filename,
                            statinfo.st_size)
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

    def which_set(self, filename, validation_percentage, testing_percentage):
        """
            Determines which data partition the file should belong to.

            We want to keep files in the same training, validation, or testing sets even
            if new ones are added over time. This makes it less likely that testing
            samples will accidentally be reused in training when long runs are restarted
            for example. To keep this stability, a hash of the filename is taken and used
            to determine which set it should belong to. This determination only depends on
            the name and the set proportions, so it won't change as other files are added.

            It's also useful to associate particular files as related (for example words
            spoken by the same person), so anything after '_nohash_' in a filename is
            ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
            'bobby_nohash_1.wav' are always in the same set, for example.

            Args:
             filename: File path of the data sample.
             validation_percentage: How much of the data set to use for validation.
             testing_percentage: How much of the data set to use for testing.

            Returns:
             String, one of 'training', 'validation', or 'testing'.
           """
        base_name = os.path.basename(filename)
        # We want to ignore anything after '_nohash_' in the file name when
        # deciding which set to put a wav in, so the data set creator has a way of
        # grouping wavs that are close variations of each other.
        hash_name = re.sub(r'_nohash_.*$', '', base_name)
        # print(type(hash_name))
        # This looks a bit magical, but we need to decide whether this file should
        # go into the training, testing, or validation sets, and we want to keep
        # existing files in the same set even if more files are subsequently
        # added.
        # To do that, we need a stable way of deciding based on just the file name
        # itself, so we do a hash of that and then use that to generate a
        # probability value that we use to assign it.
        # hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
        hash_name_hashed = hashlib.sha1(hash_name.encode()).hexdigest()
        percentage_hash = ((int(hash_name_hashed, 16) %
                            (self.MAX_NUM_WAVS_PER_CLASS + 1)) *
                           (100.0 / self.MAX_NUM_WAVS_PER_CLASS))
        if percentage_hash < validation_percentage:
            result = 'validation'
        elif percentage_hash < (testing_percentage + validation_percentage):
            result = 'testing'
        else:
            result = 'training'
        return result

    def prepare_data_index(self, silence_percentage, unknown_percentage,
                           wanted_words, validation_percentage,
                           testing_percentage):
        """
            Prepares a list of the samples organized by set and label.
            The training loop needs a list of all the available data, organized by
            which partition it should belong to, and with ground truth labels attached.
            This function analyzes the folders below the `data_dir`, figures out the
            right
            labels for each file based on the name of the subdirectory it belongs to,
            and uses a stable hash to assign it to a data set partition.
            Args:
              silence_percentage: How much of the resulting data should be background.
              unknown_percentage: How much should be audio outside the wanted classes.
              wanted_words: Labels of the classes we want to be able to recognize.
              validation_percentage: How much of the data set to use for validation.
              testing_percentage: How much of the data set to use for testing.
            Returns:
              Dictionary containing a list of file information for each set partition,
              and a lookup map for each class to determine its numeric index.
            Raises:
              Exception: If expected files are not found.
        """
        # Make sure the shuffling and picking of unknowns is deterministic.
        random.seed(self.RANDOM_SEED)
        wanted_words_index = {}
        for index, wanted_word in enumerate(wanted_words):
            wanted_words_index[wanted_word] = index + 2
        self.data_index = {'validation': [], 'testing': [], 'training': []}
        unknown_index = {'validation': [], 'testing': [], 'training': []}
        all_words = {}
        # Look through all the subfolders to find audio samples
        search_path = os.path.join(self.data_dir, '*', '*.wav')
        for wav_path in glob.glob(search_path):
            _, word = os.path.split(os.path.dirname(wav_path))
            word = word.lower()
            # Treat the '_background_noise_' folder as a special case, since we expect
            # it to contain long audio samples we mix in to improve training.
            if word == self.BACKGROUND_NOISE_DIR_NAME:
                continue
            all_words[word] = True
            set_index = self.which_set(
                wav_path, validation_percentage, testing_percentage)
            # If it's a known class, store its detail, otherwise add it to the list
            # we'll use to train the unknown label.
            if word in wanted_words_index:
                self.data_index[set_index].append(
                    {'label': word, 'file': wav_path})
            else:
                unknown_index[set_index].append(
                    {'label': word, 'file': wav_path})
        if not all_words:
            raise Exception('No .wavs found at ' + search_path)
        for index, wanted_word in enumerate(wanted_words):
            if wanted_word not in all_words:
                raise Exception('Expected to find ' + wanted_word +
                                ' in labels but only found ' +
                                ', '.join(all_words.keys()))
        # We need an arbitrary file to load as the input for the silence samples.
        # It's multiplied by zero later, so the content doesn't matter.
        silence_wav_path = self.data_index['training'][0]['file']
        for set_index in ['validation', 'testing', 'training']:
            set_size = len(self.data_index[set_index])
            silence_size = int(math.ceil(set_size * silence_percentage / 100))
            for _ in range(silence_size):
                self.data_index[set_index].append({
                    'label': self.SILENCE_LABEL,
                    'file': silence_wav_path
                })
            # Pick some unknowns to add to each partition of the data set.
            random.shuffle(unknown_index[set_index])
            unknown_size = int(math.ceil(set_size * unknown_percentage / 100))
            self.data_index[set_index].extend(
                unknown_index[set_index][:unknown_size])
        # Make sure the ordering is random.
        for set_index in ['validation', 'testing', 'training']:
            random.shuffle(self.data_index[set_index])
        # Prepare the rest of the result data structure.
        self.words_list = self.prepare_word_list(wanted_words)
        self.word_to_index = {}
        for word in all_words:
            if word in wanted_words_index:
                self.word_to_index[word] = wanted_words_index[word]
            else:
                self.word_to_index[word] = self.UNKNOWN_WORD_INDEX
        self.word_to_index[self.SILENCE_LABEL] = self.SILENCE_INDEX

    def set_size(self, mode):
        """
            Calculates the number of samples in the dataset partition.
            Args:
              mode: Which partition, must be 'training', 'validation', or 'testing'.
            Returns:
              Number of samples in the partition.
        """
        return len(self.data_index[mode])

    def prepare_background_data(self):
        """Searches a folder for background noise audio, and loads it into memory.
        It's expected that the background audio samples will be in a subdirectory
        named '_background_noise_' inside the 'data_dir' folder, as .wavs that match
        the sample rate of the training data, but can be much longer in duration.
        If the '_background_noise_' folder doesn't exist at all, this isn't an
        error, it's just taken to mean that no background noise augmentation should
        be used. If the folder does exist, but it's empty, that's treated as an
        error.
        Returns:
          List of raw PCM-encoded audio samples of background noise.
        Raises:
          Exception: If files aren't found in the folder.
        """
        self.background_data = []
        background_dir = os.path.join(
            self.data_dir, self.BACKGROUND_NOISE_DIR_NAME)
        if not os.path.exists(background_dir):
            return self.background_data

        search_path = os.path.join(self.data_dir, self.BACKGROUND_NOISE_DIR_NAME,
                                   '*.wav')
        for wav_path in glob.glob(search_path):
            wav_data, _ = librosa.load(wav_path, sr=self.sample_rate)
            self.background_data.append(wav_data)
        if not self.background_data:
            raise Exception(
                'No background wav files were found in ' + search_path)

    def get_data(self, how_many, offset, mode):
        """
            Gather samples from the data set, applying transformations as needed.
            When the mode is 'training', a random selection of samples will be returned,
            otherwise the first N clips in the partition will be used. This ensures that
            validation always uses the same samples, reducing noise in the metrics.
            Args:
              how_many: Desired number of samples to return. -1 means the entire
                contents of this partition.
              offset: Where to start when fetching deterministically.
              model_settings: Information about the current model being trained.
              background_frequency: How many clips will have background noise, 0.0 to
                1.0.
              background_volume_range: How loud the background noise will be.
              time_shift: How much to randomly shift the clips by in time.
              mode: Which partition to use, must be 'training', 'validation', or
                'testing'.
              sess: TensorFlow session that was active when processor was created.
            Returns:
              List of sample data for the transformed samples, and list of label indexes
            Raises:
              ValueError: If background samples are too short.
            """
        # Pick one of the partitions to choose samples from.
        candidates = self.data_index[mode]
        if how_many == -1:
            sample_count = len(candidates)
        else:
            sample_count = max(0, min(how_many, len(candidates) - offset))
        # Data and labels will be populated and returned.
        input_size = np.ones(self.sample_rate)
        # use GLCM features
        input_size = self.speech_features.cgram_(input_size, self.sample_rate)
        input_size = input_size.flatten()
        # print("input_size", input_size.shape)
        input_size = input_size.shape[0]
        data = np.zeros((sample_count, input_size))
        labels = np.zeros((sample_count, self.model_settings['label_count']))
        desired_samples = self.model_settings['desired_samples']
        use_background = self.background_data and (mode == 'training')
        pick_deterministically = (mode != 'training')
        time_shift = int((self.time_shift_ms * self.sample_rate) / 1000)

        # label_str = []
        # print(sample_count, data.shape, labels.shape, desired_samples, use_background, pick_deterministically)

        for i in tqdm.tqdm(range(offset, offset + sample_count)):
            # for i in range(offset, offset + sample_count):
            if how_many == -1 or pick_deterministically:
                sample_index = i
            else:
                sample_index = np.random.randint(len(candidates))

            sample = candidates[sample_index]
            # print(sample)
            # time shifting setting
            if time_shift > 0:
                time_shift_amount = np.random.randint(-time_shift, time_shift)
            else:
                time_shift_amount = 0
            if time_shift_amount > 0:
                time_shift_padding = [time_shift_amount, 0]
                time_shift_offset = 0
            else:
                time_shift_padding = [0, -time_shift_amount]
                time_shift_offset = -time_shift_amount

            # print(time_shift_padding, time_shift_offset, time_shift_offset)

            # choose background to mix in
            if use_background or sample['label'] == self.SILENCE_LABEL:
                background_index = np.random.randint(len(self.background_data))
                background_samples = self.background_data[background_index]
                if len(background_samples) <= self.model_settings['desired_samples']:
                    raise ValueError(
                        'Background sample is too short! Need more than %d'
                        ' samples but only %d were found' %
                        (self.model_settings['desired_samples'], len(
                            background_samples))
                    )
                background_offset = np.random.randint(
                    0, len(background_samples) -
                       self.model_settings['desired_samples']
                )
                background_clipped = background_samples[background_offset:(
                        background_offset + desired_samples
                )]
                background_reshaped = background_clipped.reshape(
                    desired_samples)
                if sample['label'] == self.SILENCE_LABEL:
                    background_volume = np.random.uniform(0, 1)
                elif np.random.uniform(0, 1) < self.background_frequency:
                    background_volume = np.random.uniform(
                        0, self.background_volume)
                else:
                    background_volume = 0
            else:
                background_reshaped = np.zeros(desired_samples)
                background_volume = 0

            # print(background_reshaped.shape, background_volume)
            # librosa.output.write_wav('data/junk/bg_resh.wav', background_reshaped, self.sample_rate)
            # if we want silence, mute out the main sample but leave the background
            if sample['label'] == self.SILENCE_LABEL:
                foreground_volume = 0
            else:
                foreground_volume = 1

            input_data = {
                'wav_filename': sample['file'],
                'time_shift_padding': time_shift_padding,
                'time_shift_offset': time_shift_offset,
                'background_data': background_reshaped,
                'background_volume': background_volume,
                'foreground_volume': foreground_volume
            }
            # get processed audio
            wav_result = self.audio_util.processing_audio(
                input_data, normalize=True)
            # feature extraction
            feature = self.speech_features.cgram_(wav_result, 16000)
            # feature = self.speech_features.mfcc(wav_result)
            # feature = self.speech_features.mfcc_psf(wav_result)
            feature = feature.flatten()
            # print("feature shape: ", feature.shape)
            # data.append(feature)
            data[i - offset, :] = feature
            label_index = self.word_to_index[sample['label']]
            # label_str.append(sample['label'])
            labels[i - offset, label_index] = 1
            # np.set_printoptions(threshold=np.nan)
            # print(labels)

        return data, labels

    def label_transform(self, labels):
        n_labels = []
        print(self.words_list)
        for label in labels:
            if label == '_silence_':
                n_labels.append('_silence_')
            elif label not in self.wanted_words:
                n_labels.append('_unknown_')
            else:
                n_labels.append(str(label))
        return np.array(n_labels)

    def integer_encoding(self, labels):
        label_encoder = LabelEncoder()
        return label_encoder.fit_transform(self.label_transform(labels))

    def onehot_encoder(self, labels):
        onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
        integer_encoded = labels.reshape(len(labels), 1)
        onehot = onehot_encoder.fit_transform(integer_encoded)
        return onehot
