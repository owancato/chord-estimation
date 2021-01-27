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
from read_data import train_y,output_dim

model = keras.models.load_model('DNN_v2')
newinput=model.get_layer(index=14).output

H = 200
model2 = models.Sequential()

model2.add(layers.Dense(H))
model2.add(layers.BatchNormalization())
model2.add(layers.Activation(activations.relu))
model2.add(layers.Dropout(0.3))

model2.add(layers.Dense(H))
model2.add(layers.BatchNormalization())
model2.add(layers.Activation(activations.relu))
model2.add(layers.Dropout(0.3))

model2.add(layers.Dense(H))
model2.add(layers.BatchNormalization())
model2.add(layers.Activation(activations.relu))
model2.add(layers.Dropout(0.3))

model2.add(layers.Dense(H))
model2.add(layers.BatchNormalization())
model2.add(layers.Activation(activations.relu))
model2.add(layers.Dropout(0.3))



model2.add(layers.Dense(output_dim, activation='softmax'))

adam = tf.keras.optimizers.Adadelta(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
history = model.fit(newinput, train_y, epochs=18, batch_size= 2000)

model.save('DNN_v3')