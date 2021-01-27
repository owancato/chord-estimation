import json
import pandas as pd
import numpy as np
import librosa
import librosa.display
import os
import csv
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import datasets, layers, models, activations
import tensorflow as tf



with open(r'CE200/1/feature.json') as f:
    data = json.load(f)

col_name = ['chroma_stft',
 'chroma_cqt',
 'chroma_cens',
 'rms',
 'spectral_centroid',
 'spectral_bandwidth',
 'spectral_contrast',
 'spectral_flatness',
 'spectral_rolloff',
 'poly_features',
 'tonnetz',
 'zero_crossing_rate']

col_name = ['chroma_cqt']


def get_feature(song_id):
    all_ls = []

    with open(r'CE200/' + str(song_id) + '/feature.json') as f:
        data = json.load(f)

    ls = []
    for key in col_name:
        ls.append(np.array(data[key]))
    data = np.concatenate(ls, axis=0)

    length = data.shape[1]
    # print(length)
    return data.T, length

def get_label(song_id, length):
    with open (r"CE200/"+str(song_id)+"/ground_truth.txt", 'r') as f:
        label = [column for column in csv.reader(f,delimiter='\t')]
        vector = [0] * length
        #print(len(vector))
        for i in label:
            start = round(float(i[0])/(1/22050*512))
            end = round(float(i[1])/(1/22050*512))
            if end > length:
                end = length
            for j in range(start,end):
                vector[j] = i[2]
        #print(len(vector))
        for i in range(length):
            if vector[i] == 0:
                vector[i] = 'N'
    #print(len(vector))
    return vector

chordLabel = ['N','C:maj','D:maj','E:maj','F:maj','G:maj','A:maj','B:maj','C#:maj','Db:maj','D#:maj','Eb:maj','F#:maj','Gb:maj','G#:maj','Ab:maj','A#:maj','Bb:maj','C:min','D:min','E:min','F:min','G:min','A:min','B:min','C#:min','Db:min','D#:min','Eb:min','F#:min','Gb:min','G#:min','Ab:min','A#:min','Bb:min','C:maj7','D:maj7','E:maj7','F:maj7','G:maj7','A:maj7','B:maj7','C#:maj7','Db:maj7','D#:maj7','Eb:maj7','F#:maj7','Gb:maj7','G#:maj7','Ab:maj7','A#:maj7','Bb:maj7','C:min7','D:min7','E:min7','F:min7','G:min7','A:min7','B:min7','C#:min7','Db:min7','D#:min7','Eb:min7','F#:min7','Gb:min7','G#:min7','Ab:min7','A#:min7','Bb:min7','C:7','D:7','E:7','F:7','G:7','A:7','B:7','C#:7','Db:7','D#:7','Eb:7','F#:7','Gb:7','G#:7','Ab:7','A#:7','Bb:7']
label_encoder = OneHotEncoder(handle_unknown='ignore')
label_encoder.fit(pd.DataFrame(chordLabel))

label_encoder.get_feature_names()


def get_all(song_list, k=3, a=1):
    x_ls = []
    y_ls = []

    frames = int(((2 * k) / a) + 1)
    k_ls = range(-k, k + 1, a)

    for song in song_list:
        # print(song)
        raw, l = get_feature(song)
        row_num = raw.shape[0]
        col_num = raw.shape[1]
        x = np.zeros((row_num, col_num * frames))

        for row in range(k):
            for frame in range(frames):
                if row + k_ls[frame] < 0:
                    x[row, col_num * frame:col_num * (frame + 1)] = raw[0]
                else:
                    x[row, col_num * frame:col_num * (frame + 1)] = raw[row + k_ls[frame]]

        for row in range(k, row_num - k):
            for frame in range(frames):
                if row + k_ls[frame] > row_num - 1:
                    x[row, col_num * frame:col_num * (frame + 1)] = raw[-1]
                else:
                    x[row, col_num * frame:col_num * (frame + 1)] = raw[row + k_ls[frame]]

        for frame in range(frames):
            if row + k_ls[frame] >= row_num:
                x[row, col_num * frame:col_num * (frame + 1)] = raw[-1]
            else:
                x[row, col_num * frame:col_num * (frame + 1)] = raw[row + k_ls[frame]]

        x_ls.append(x)
        y_ls += get_label(song, row_num)
        # print('\n')

    data_x = np.concatenate(x_ls, axis=0)
    data_y = pd.DataFrame(y_ls)

    encoded_data_y = label_encoder.transform(data_y)

    return data_x, encoded_data_y.A

train_x, train_y = get_all(range(1,181), k = 12, a = 4)

valid_x, valid_y = get_all(range(181,200), k = 12, a = 4)

test_x, test_y = get_all(range(181,200), k = 12, a = 4)

input_dim = train_x.shape[1]
output_dim = train_y.shape[1]
print('input_dim',train_x)
print('output_dim',output_dim)

H = 200
model = models.Sequential()

model.add(layers.Dense(H, input_dim=input_dim))
model.add(layers.BatchNormalization())
model.add(layers.Activation(activations.relu))
model.add(layers.Dropout(0.3))

model.add(layers.Dense(H))
model.add(layers.BatchNormalization())
model.add(layers.Activation(activations.relu))
model.add(layers.Dropout(0.3))

model.add(layers.Dense(H))
model.add(layers.BatchNormalization())
model.add(layers.Activation(activations.relu))
model.add(layers.Dropout(0.3))

model.add(layers.Dense(H))
model.add(layers.BatchNormalization())
model.add(layers.Activation(activations.relu))
model.add(layers.Dropout(0.3))



model.add(layers.Dense(output_dim, activation='softmax'))

adam = tf.keras.optimizers.Adadelta(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
history = model.fit(train_x, train_y, epochs=15, batch_size= 2000,
                    validation_data=(valid_x, valid_y))

model.save('DNN_v2')