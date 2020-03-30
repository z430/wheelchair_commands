import input_data
import numpy as np
import datetime
import signal
import sys
import random  # feature versioning

t = datetime.datetime.now()
newdate = datetime.datetime.strftime(t, "%H_%M_%m_%d")
sr = random.SystemRandom()
version_number = sr.getrandbits(12)


def save_experiment_data():
    """
    saving experiment data is not storage efficient, but it is really useful when
    we only experiment with the model not the data.
    """
    wanted_words = 'left,right,forward,backward,stop,go'
    Data = input_data.GetData(wanted_words=wanted_words)
    print(Data.validation_percentage)
    Data.initialize()

    feature_name = 'glcm4_spec_{}'.format(version_number)
    print(feature_name)

    print("generating training, validation, testing data.")
    x_train, y_train = Data.get_data(-1, 0, 'training')
    x_val, y_val = Data.get_data(-1, 0, 'validation')
    x_test, y_test = Data.get_data(-1, 0, 'testing')

    training_size = Data.set_size('training')
    testing_size = Data.set_size('testing')
    print(training_size, testing_size)

    print("saving the data")
    np.save("data/junk/x_train_{}.npy".format(feature_name), x_train)
    np.save("data/junk/y_train_{}.npy".format(feature_name), y_train)

    np.save("data/junk/x_val_{}.npy".format(feature_name), x_val)
    np.save("data/junk/y_val_{}.npy".format(feature_name), y_val)

    np.save("data/junk/x_test_{}.npy".format(feature_name), x_test)
    np.save("data/junk/y_test_{}.npy".format(feature_name), y_test)


def signal_handler(self, sig, frame):
    print("CTRL + C Detected")
    sys.exit()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    save_experiment_data()
    signal.pause()
    # np.set_printoptions(threshold=np.nan)
    # y_train = np.load("data/junk/y_train_10_22_01_09.npy")
    # y_test = np.load("data/junk/y_test_10_22_01_09.npy")
    # print(y_train)
    # wanted_words = 'yes,no,up,down,left,right,on,off,stop,go'
    # data = input_data.GetData(prepare_data=False, wanted_words=wanted_words)
    # print(data.integer_encoding(y_train))
    # svm_glcm_matrix()
