import json
import pandas as pd
import numpy as np
import librosa
import librosa.display
import os
import csv
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import datasets, layers, models, activations
from tensorflow import keras
import tensorflow as tf

model = keras.models.load_model('DNN_v2')


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


def get_all(song_list, k=3, a=1):
    x_ls = []
    length_ls = []
    frames = int(((2 * k) / a) + 1)
    k_ls = range(-k, k + 1, a)

    for song in song_list:
        # print(song)
        raw, l = get_feature(song)
        length_ls.append(l)
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
    data_x = np.concatenate(x_ls, axis=0)

    return data_x, length_ls

def get_output(y, length_ls, id):
    y = np.argmax(y, axis=-1)
    all_ls = []
    name_ls = []
    count = 0
    song_number = len(length_ls)

    if len(id) != len(length_ls):
        print('shit')

    for i in range(song_number):
        if i == 0:
            ls = get_output_list(y[0:length_ls[i]])
        else:
            ls = get_output_list(y[count: count + length_ls[i]])
        count += length_ls[i]
        all_ls.append(ls)
        name_ls.append(str(id[i]))

    return dict(zip(name_ls,all_ls))


def get_output_list(y):
    ls = []
    startTime = 0
    endTime = 0
    last = 0
    lastchord = 'N'
    tmp = 0
    first = 1
    nowchord = 0
    end2 = 0
    for i in range(len(y)):
        if (last != y[i]):
            endTime = i + 1
            t1 = (startTime / 22050) * 512
            t2 = (endTime / 22050) * 512
            if t2 - t1 > 0.3:
                ls.append([t1, t2, label_list[y[i - 1]][3:]])
                startTime = i + 1
                lastchord = label_list[y[i - 1]][3:]
                tmp = 0
            elif label_list[y[i - 1]][3:] == 'N' and first == 0:
                first = 0
                tmp = 0
                startTime = i + 1
                lastchord = label_list[y[i - 1]][3:]
                ls.append([t1, t2, label_list[y[i - 1]][3:]])
            else:
                ls.append([t1, t2, lastchord])
                startTime = i + 1
        end2 = i + 1
        nowchord = label_list[y[i]][3:]
        last = y[i]
    ls.append([(startTime / 22050) * 512, (end2 / 22050) * 512, nowchord])
    newls = []
    last = ls[0][2]
    nowchord = 0
    startTime = 0
    endTime = 0
    end2 = 0
    for i in range(len(ls)):
        if last != ls[i][2]:
            t1 = startTime
            t2 = ls[i][0]
            newls.append([t1, t2, last])
            startTime = ls[i][0]
        last = ls[i][2]
        end2 = ls[i][1]
        nowchord = ls[i][2]
    newls.append([startTime, end2, nowchord])

    return newls

def decodeoutput(song_id,dic):
    length=round(float(dic[str(song_id)][-1][1])/ (1 / 22050 * 512))
    label=[column for column in dic[str(song_id)]]
    vector = [0] * length
    for i in label:
        start = round(float(i[0]) / (1 / 22050 * 512))
        end = round(float(i[1]) / (1 / 22050 * 512))
        if end > length:
            end = length
        for j in range(start, end):
            vector[j] = i[2]
        # print(len(vector))
        for i in range(length):
            if vector[i] == 0:
                vector[i] = 'N'
    return vector

def get_label(song_id):
    with open (r"CE200/"+str(song_id)+"/ground_truth.txt", 'r') as f:
        label = [column for column in csv.reader(f,delimiter='\t')]
        length=round(float(label[-1][1])/(1/22050*512))
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

def decode_all(dic):
    pred=[]
    ref=[]
    for song_id in range(181,200):
        newls=decodeoutput(song_id,dic)
        pred.append(newls)
        tmpls=get_label(song_id)
        ref.append((tmpls))
    return pred,ref

def get_score(pred,ref):
    correct=0
    total=0
    for i in range(len(pred)):
        print(len(pred[i]),len(ref[i]))
        if len(pred[i])<=len(ref[i]):
            for j in range(len(pred[i])):
                if pred[i][j]==ref[i][j]:
                    correct=correct+1
                total=total+1
        else:
            for j in range(len(ref[i])):
                if pred[i][j]==ref[i][j]:
                    correct=correct+1
                total=total+1
    score=(correct/total)*100
    return score


col_name = ['chroma_cqt']

chordLabel = ['N','C:maj','D:maj','E:maj','F:maj','G:maj','A:maj','B:maj','C#:maj','Db:maj','D#:maj','Eb:maj','F#:maj','Gb:maj','G#:maj','Ab:maj','A#:maj','Bb:maj','C:min','D:min','E:min','F:min','G:min','A:min','B:min','C#:min','Db:min','D#:min','Eb:min','F#:min','Gb:min','G#:min','Ab:min','A#:min','Bb:min','C:maj7','D:maj7','E:maj7','F:maj7','G:maj7','A:maj7','B:maj7','C#:maj7','Db:maj7','D#:maj7','Eb:maj7','F#:maj7','Gb:maj7','G#:maj7','Ab:maj7','A#:maj7','Bb:maj7','C:min7','D:min7','E:min7','F:min7','G:min7','A:min7','B:min7','C#:min7','Db:min7','D#:min7','Eb:min7','F#:min7','Gb:min7','G#:min7','Ab:min7','A#:min7','Bb:min7','C:7','D:7','E:7','F:7','G:7','A:7','B:7','C#:7','Db:7','D#:7','Eb:7','F#:7','Gb:7','G#:7','Ab:7','A#:7','Bb:7']
label_encoder = OneHotEncoder(handle_unknown='ignore')
label_encoder.fit(pd.DataFrame(chordLabel))
label_list = label_encoder.get_feature_names()

test_x, length_ls = get_all(range(181,200),k = 12, a =4)
y = model.predict(test_x)
dic = get_output(y, length_ls, range(181,200))
pred,ref=decode_all(dic)
print(pred[0:500])
print(ref[0:500])
print(len(pred))
print(len(ref))
score=get_score(pred,ref)
print(score)

