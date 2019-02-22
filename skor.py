import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Activation, Dense
from keras.optimizers import SGD
import matplotlib.pylab as plt
from keras.utils.np_utils import to_categorical
from keras.models import load_model


def scores(a,b):
    return a+b
