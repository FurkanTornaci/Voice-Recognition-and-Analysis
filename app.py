# app.py
from ast import Return
from email import message
from traceback import print_tb
from flask import Flask, request, render_template
import os
import librosa
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dropout, Activation, Conv1D, MaxPooling1D, Dense
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
from matplotlib.pyplot import figure


import io
from matplotlib.figure import Figure
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from flask import Flask




import json




app = Flask(__name__)

wsgi_app = app.wsgi_app

UPLOAD_FOLDER = 'static/uploads/'



def features_extractor(audio,sample_rate):
    
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    
    return mfccs_scaled_features

def split_into_intervals(data, k = 1600):
    intervals = [data[n*k:(n+1)*k] for n in range(int(len(data)/k))]
    return intervals

@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        FileName = file.filename
        file.save(os.path.join("static", file.filename))
        return render_template('index.html', message="success")
    return render_template('index.html', message="Upload")


@app.route('/hi', methods=["GET", "POST"])
def hi():
    model = keras.models.load_model('./saved_models/audio_classification.hdf5')
    features = []
    
    data, sr = librosa.load("./d-othervoice65I.wav", sr=8000)
    intervals = split_into_intervals(np.array(data), 1600)
    for i in intervals:
        features.append(features_extractor(i,8000))
    prediction = (model.predict(np.array(features)) > 0.5 ) + 0
    print(prediction)
    child = []
    nonChild = [] 
    for i in range(len(prediction)):
        if prediction[i][0] == 1:
            child.extend(intervals[i])
            nonChild.extend(np.zeros(len(intervals[i])))
        else:
            nonChild.extend(intervals[i])
            child.extend(np.zeros(len(intervals[i])))


    total = len(nonChild)
    fig = plt.figure(figsize=(16, 12))
    plt.title("Signal Wave...")
    plt.plot(np.array(child))
    plt.plot(np.array(nonChild),color = "orange")
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')



if __name__ == "__main__":
    import os
    app.run()
