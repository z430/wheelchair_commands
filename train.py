""" Training """

import random
import time

import keras
import numpy as np
import tensorflow as tf
from keras import losses, optimizers
import pandas as pd

import input_data

def main():
    """ ------------------- Features Configuration ------------------- """
    wanted_words = 'left,right,forward,backward,stop,go'.split(",")
    print(type(wanted_words))
    features = input_data.GetData(
        wanted_words=wanted_words, feature="mfcc")

    all_files = features.get_datafiles("training")

    # transform the list dicts into dataframe
    training_data = pd.DataFrame(all_files)
    print(training_data.head())
    print(training_data["label"].unique())


if __name__ == '__main__':
    main()
