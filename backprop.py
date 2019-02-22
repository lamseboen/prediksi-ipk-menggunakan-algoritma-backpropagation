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
from keras import backend as K
from sklearn import metrics

def read_file(filename):
    file = pd.read_excel(filename)
    file = file.values.tolist()
    return file

def load_data(filename):
    df = pd.read_excel(filename)
    X = np.array(df.drop(['jenis_kelamin', 'semester', 'ipk', 'target','keterangan'], axis=1))
    Y = np.array(df['target'])
    return X, Y



def train_test(x, y, train_split=0.8):
    TRAIN_SPLIT = train_split
    X, Y = shuffle(x, y)
    X_train, X_test = X[:int(TRAIN_SPLIT * len(X))], X[int(TRAIN_SPLIT * len(X)):]
    Y_train, Y_test = Y[:int(TRAIN_SPLIT * len(Y))], Y[int(TRAIN_SPLIT * len(Y)):]

    return X_train, X_test, Y_train, Y_test

def train_test2(x, y, train_split=0.8):
    TRAIN_SPLIT = train_split
    X, Y = shuffle(x, y)
    X_train, X_test = X[:int(TRAIN_SPLIT * len(X))], X[int(TRAIN_SPLIT * len(X)):]
    Y_train, Y_test = Y[:int(TRAIN_SPLIT * len(Y))], Y[int(TRAIN_SPLIT * len(Y)):]
    Y_test2 = Y_test.copy()
    return X_train, X_test, Y_train, Y_test, Y_test2


#convert label to categorical
def label_cat(Y_train, Y_test):
    Y_train = to_categorical(Y_train, num_classes=None)
    Y_test = to_categorical(Y_test, num_classes=None)
    return Y_train, Y_test

def train(x, y, learning_rate=0.04, batch_size=7, n_epochs=1000, layerx=5,aktiv1="relu"):
    np.random.seed(7)
    X_train, X_test, Y_train, Y_test = train_test(x, y)
    Y_train, Y_test = label_cat(Y_train,Y_test)


    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=7, activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(6, activation='softmax'))

    # Compile model
    sgd = SGD(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # Fit the model
    backprop = model.fit(X_train, Y_train, epochs=n_epochs, batch_size=batch_size, verbose=2)


    scores = model.evaluate(X_test, Y_test)
    model.save('./media/model/model_backprop.h5')

    matriks = model.metrics_names[1]
    skor = scores[1]*100
    K.clear_session()
    return backprop, matriks, skor

def predict(data):
    y_dict = {
        1 : 'Kurang',
        2 : 'Cukup',
        3 : 'Baik',
        4 : 'Memuaskan',
        5 : 'Cumlaude'
    }


def train2(x, y, learning_rate=0.3, batch_size=50, n_epochs=1000, layerx = [10,10,5], aktivx = ['relu', 'relu', 'softmax']):
    np.random.seed(7)
    X_train, X_test, Y_train, Y_test, Y_test2 = train_test2(x, y)
    Y_train, Y_test = label_cat(Y_train,Y_test)




    # create model
    model = Sequential()
    model.add(Dense(int(layerx[0]), input_dim=7, activation=aktivx[0]))
    for idx, val in enumerate(layerx):
        if idx == 0:
            temp = 0
        else:
            model.add(Dense(int(layerx[idx]), activation=aktivx[idx]))
    model.add(Dense(6, activation='softmax'))
    # model.add(Dense(int(layerx[2]), activation=aktivx[2]))
    # model.add(Dense(5, activation='relu'))
    # model.add(Dense(6, activation='softmax'))

    # Compile model
    sgd = SGD(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # Fit the model
    backprop = model.fit(X_train, Y_train, epochs=n_epochs, batch_size=batch_size, verbose=2)
    print(aktivx)
    print(layerx)

    scores = model.evaluate(X_test, Y_test)
    model.save('./media/model/model_backprop.h5')

    matriks = model.metrics_names[1]
    skor = scores[1]*100
    #print (metrics.classification_report(Y_test, clf.predict(X_test)))
   # print ("accuracy:" ,metrics.accuracy_score(Y_test, clf.predict(X_test)))

    hasil = model.predict(X_test)
    confus_dict = {
        1 : 1,
        2 : 2,
        3 : 3,
        4 : 4,
        5 : 5
    }

    index_max = np.argmax(hasil[21])
    predicted = []
    for i in range(len(hasil)):
        index = np.argmax(hasil[i])
        predicted.append(confus_dict[index])

    print(predicted)
    print(Y_test2)
    confus_matrix = metrics.classification_report(Y_test2, predicted, labels=[1, 2, 3, 4, 5])
    print(confus_matrix)


    K.clear_session()
    return backprop, matriks, skor

def predict(data):
    y_dict = {
        1 : 'Kurang',
        2 : 'Cukup',
        3 : 'Baik',
        4 : 'Memuaskan',
        5 : 'Cumlaude'
    }


# if __name__ == '__main__':
#     file = 'training.xlsx'
#     x, y = load_data(file)
#     backprop, matriks, skor = train(x, y)
#     print(print("\n%s: %.2f%%" % (matriks, skor)))
    # df = pd.read_excel(file)
    # print(df)
