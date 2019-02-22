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
from backprop import *
from keras import backend as K

def load_data_prediksi(filename):
    df = pd.read_excel(filename)
    X = np.array(df)
    return X

score = 0
empty = False
def scoress(filemodel, filedata):
    x,y = load_data(filedata);
    X_train, X_test, Y_train, Y_test = train_test(x, y)
    Y_train, Y_test = label_cat(Y_train, Y_test)
    #model = load_model(filemodel)
    model = load_model("./media/model/model_backprop.h5")
    scores = model.evaluate(X_test, Y_test)
    skor = scores[1]*100
    K.clear_session()
    return skor

def prediksi_jwb(x1,x2,x3,x4,x5,x6,x7):
    list = [[x1,x2,x3,x4,x5,x6,x7]]
    X_test = np.asarray(list)
    model = load_model("./media/model/model_backprop.h5")
    y_dict = {
        1 : 'Kurang',
        2 : 'Cukup',
        3 : 'Baik',
        4 : 'Memuaskan',
        5 : 'Cumlaude'
        }
    hasil = model.predict(X_test)
    for i in range(len(hasil)):
        index = np.argmax(hasil[i])
        predicted = y_dict[index]
    K.clear_session()
    return predicted

def prediksi_jwb_file(filedata):
    hasil_prediksi = []
    data = load_data_prediksi(filedata)
    X_test = data
    model = load_model("./media/model/model_backprop.h5")
    y_dict = {
        1 : 'Kurang',
        2 : 'Cukup',
        3 : 'Baik',
        4 : 'Memuaskan',
        5 : 'Cumlaude'
        }
    hasilnya = model.predict(X_test)
    for i in range(len(hasilnya)):
        index = np.argmax(hasilnya[i])
        predicted = y_dict[index]
        hasil_prediksi.append(predicted)
    K.clear_session()
    return hasil_prediksi

def hitung(hasil):
    jml_memuaskan = 0
    jml_cumlaude = 0
    jml_baik = 0
    jml_cukup = 0
    jml_kurang = 0
    for i in hasil:
        if i == "Memuaskan":
            jml_memuaskan = jml_memuaskan+1
        elif i == "Cumlaude":
            jml_cumlaude = jml_cumlaude+1
        elif i == "Baik":
            jml_baik = jml_baik+1
        elif i == "Cukup":
            jml_cukup = jml_cukup+1
        elif i == "Kurang":
            jml_kurang = jml_kurang+1
    return jml_memuaskan, jml_cumlaude, jml_baik, jml_cukup, jml_kurang
