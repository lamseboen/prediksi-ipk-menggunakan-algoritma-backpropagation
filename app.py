import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug import secure_filename
from keras.models import load_model
from backprop import *
from backprop2 import *
import tensorflow as tf
import json



UPLOAD_FOLDER_TRAINING = './media/data_training'
UPLOAD_FOLDER_TESTING = './media/data_testing'
UPLOAD_FOLDER_MODEL = './media/model'
UPLOAD_FOLDER_PREDIKSI = './media/data_prediksi'

nama_training = "training.xlsx"
nama_testing = "testing.xlsx"
nama_model = "model_backprop.h5"
dir_training = "./media/data_training/"
dir_testing = "./media/data_testing/"
dir_model = "./media/model/"

app = Flask(__name__)
app.config['UPLOAD_FOLDER_TRAINING'] = UPLOAD_FOLDER_TRAINING
app.config['UPLOAD_FOLDER_TESTING'] = UPLOAD_FOLDER_TESTING
app.config['UPLOAD_FOLDER_MODEL'] = UPLOAD_FOLDER_MODEL
app.config['UPLOAD_FOLDER_PREDIKSI'] = UPLOAD_FOLDER_PREDIKSI

@app.route('/')
@app.route('/home')
def index():
    return render_template('home.html')


# @app.route('/data_training')
# def data_training():
#     data_training = read_file(dir_training+nama_training)
#     jumlah_data = len(data_training)
#     return render_template('view_data_training.html', data_training = data_training, jumlah_data=jumlah_data)

####data testing###
#@app.route('/data_testing')
#def data_testing():
   # data_testing = read_file(dir_testing+nama_testing)
   # jumlah_data = len(data_testing)
   # return render_template('view_data_testing.html', data_testing = data_testing, jumlah_data=jumlah_data)


@app.route('/data_training', methods = ['GET', 'POST'])
def upload_training():
    if request.method == 'POST':
        if 'upload_training' in request.form:
            filelist = [ f for f in os.listdir(dir_training) if f.endswith(".xlsx") ]
            for f in filelist:
                os.remove(os.path.join(dir_training, f))

            f = request.files['filetraining']
            filename = secure_filename(f.filename)
            f.save(os.path.join(app.config['UPLOAD_FOLDER_TRAINING'], filename))
            nama_awal = filename
            nama_baru = nama_training
            os.rename(dir_training+nama_awal,dir_training+nama_baru)
            training = read_file(dir_training+nama_baru)

            return redirect(url_for('upload_training'))
    data_training = read_file(dir_training+nama_training)
    jumlah_data = len(data_training)
    return render_template('view_data_training.html', data_training = data_training, jumlah_data=jumlah_data)
    #return render_template('view_data_training.html', data_training = data_training)

        #if 'upload_testing' in request.form:
         #   filelist = [ f for f in os.listdir(dir_testing) if f.endswith(".xlsx") ]

          #  for f in filelist:
           #    os.remove(os.path.join(dir_testing, f))

            #f = request.files['filetesting']
            #filename = secure_filename(f.filename)
            #f.save(os.path.join(app.config['UPLOAD_FOLDER_TESTING'], filename))
            #nama_awal = filename
            #nama_baru = nama_testing
            #os.rename(dir_testing+nama_awal,dir_testing+nama_baru)
            #testing = read_file(dir_testing+nama_baru)


            #return 'file uploaded successfully {}'.format(testing)
            #return redirect(url_for('data_testing'))





@app.route('/prediksi',  methods = ['GET', 'POST'])
def prediksi():
    if request.method == "POST":
        if 'upload_prediksi' in request.form:
            filelist = [ f for f in os.listdir('./media/data_prediksi') if f.endswith(".xlsx") ]
            for f in filelist:
                os.remove(os.path.join('./media/data_prediksi', f))

            f = request.files['fileprediksi']
            filename = secure_filename(f.filename)
            f.save(os.path.join(app.config['UPLOAD_FOLDER_PREDIKSI'], filename))
            nama_awal = filename
            nama_baru = nama_testing
            os.rename('./media/data_prediksi/'+filename,'./media/data_prediksi/prediksi.xlsx')
            testing = read_file('./media/data_prediksi/prediksi.xlsx')
            hasil_prediksi = prediksi_jwb_file('./media/data_prediksi/prediksi.xlsx')
            print (hasil_prediksi)
            memuaskan, cumlaude, baik, cukup, kurang = hitung(hasil_prediksi)
            data_grafik = [memuaskan, cumlaude, baik, cukup, kurang]
            print(data_grafik)
            #return render_template('test.html')
            return render_template('test.html', hasil_prediksi = hasil_prediksi, data_grafik = data_grafik )
        if 'prediksi' in request.form:
            f = request.form
            x1 = f['x1']
            x2 = f['x2']
            x3 = f['x3']
            x4 = f['x4']
            x5 = f['x5']
            x6 = f['x6']
            x7 = f['x7']
            hasil = prediksi_jwb(x1,x2,x3,x4,x5,x6,x7)
            #return "hasil {}".format(hasil)
            return  render_template('hasil.html', hasil = hasil)


    file = './media/data_training/training.xlsx'
    model = "./media/model/model_backprop.h5"
    #x, y = load_data(file)
    score = scoress(model, file)
    name = score
    #return "hasil {}".format(hasil)
    return render_template('prediksi.html',  name = name)

@app.route('/backpropagation', methods = ['GET', 'POST'])
def backpropagation():
    if request.method == 'POST':
        h_l = []
        a_l = []
        f = request.form
        print(len(f))
        lr = float(f['learning_rate'])
        learning_rate = float(lr)
        epo = f['epochs']
        epoch = int(epo)
        h_l = f.getlist('hl[]')
        a_l = f.getlist('al[]')

        file = './media/data_training/training.xlsx'
        x, y = load_data(file)
        #X_train, X_test, Y_train, Y_test = train_test(x, y)
        #backprop, matriks, skor = train(x, y, learning_rate = lr, n_epochs = epoch)
        backprop, matriks, skor = train2(x, y, learning_rate = lr, n_epochs = epoch, layerx= h_l, aktivx = a_l)#, learning_rate = learning_rate, n_epochs = epochs)
        return "Sukses kakak, learning_rate = {},epoch = {},  akurasi = {} ".format(lr,epoch,  skor)

    return render_template('backpropagation.html')





if __name__ == '__main__':
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True)
