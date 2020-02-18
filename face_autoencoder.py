from __future__ import absolute_import, division, print_function

from tensorflow import keras

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import os
import cv2

PATHS_CSV = "/Users/james.neve/Data/Attractiveness/LikeCounts/pathsdata.csv"
TARGET_CSV = "/Users/james.neve/Data/Attractiveness/FaceSamplesBQViews2LExtracted/pageview_counts_20200129.csv"
IMAGES_DIR = "/Users/james.neve/Data/Attractiveness/FaceSamplesBQViews2L/"
FACES_DIR = "/Users/james.neve/Data/Attractiveness/FaceSamplesBQViews2LExtracted/"
#FACES_DIR = "/Users/james.neve/Data/Attractiveness/FaceSamplesM/"
QUERY_RES = "/Users/james.neve/Data/Attractiveness/LikeCounts/time_normalised_data.csv"

FACE_WIDTH = 96
FACE_HEIGHT = 96

face_files = os.listdir(FACES_DIR)

print('Processing training data')

train_faces = np.empty([len(face_files), FACE_WIDTH, FACE_HEIGHT, 3])

i = 0
for _, face_file in enumerate(face_files):
    if face_file[-4:] != ".jpg":
        continue
    face_path = "%s%s" % (FACES_DIR, face_file)

    face = cv2.imread(face_path, cv2.IMREAD_COLOR)
    if np.shape(face) != (FACE_HEIGHT, FACE_WIDTH, 3):
        continue

    for j in [0, 1, 2]:
        train_faces[i, j] = face[j, :, :]
    i += 1

train_faces = train_faces[:i]

print('Normalising RGB')

train_faces = train_faces / 255.0

autoencoder = keras.Sequential()

# Encode
autoencoder.add(keras.layers.Conv2D(3, (7, 7), input_shape=(FACE_WIDTH, FACE_HEIGHT, 3), activation="relu", padding="same"))
autoencoder.add(keras.layers.MaxPooling2D(pool_size=(3, 3), padding="same"))
autoencoder.add(keras.layers.BatchNormalization())

autoencoder.add(keras.layers.Conv2D(64, (1, 1), activation="relu", padding="same"))
autoencoder.add(keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
autoencoder.add(keras.layers.BatchNormalization())
autoencoder.add(keras.layers.MaxPooling2D(pool_size=(4, 4), padding="same"))

autoencoder.add(keras.layers.Conv2D(192, (1, 1), activation="relu", padding="same"))
autoencoder.add(keras.layers.Conv2D(192, (3, 3), activation="relu", padding="same"))
autoencoder.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same"))

autoencoder.add(keras.layers.Conv2D(384, (1, 1), activation="relu", padding="same"))
autoencoder.add(keras.layers.Conv2D(384, (3, 3), activation="relu", padding="same"))

# Decode
autoencoder.add(keras.layers.Conv2D(384, (3, 3), activation="relu", padding="same"))
autoencoder.add(keras.layers.Conv2D(384, (1, 1), activation="relu", padding="same"))

autoencoder.add(keras.layers.UpSampling2D((2, 2)))
autoencoder.add(keras.layers.Conv2D(192, (3, 3), activation="relu", padding="same"))
autoencoder.add(keras.layers.Conv2D(192, (1, 1), activation="relu", padding="same"))

autoencoder.add(keras.layers.UpSampling2D((4, 4)))
autoencoder.add(keras.layers.BatchNormalization())
autoencoder.add(keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
autoencoder.add(keras.layers.Conv2D(64, (1, 1), activation="relu", padding="same"))

autoencoder.add(keras.layers.BatchNormalization())
autoencoder.add(keras.layers.UpSampling2D((3, 3)))
autoencoder.add(keras.layers.Conv2D(3, (7, 7), activation="sigmoid", padding="same"))

autoencoder.summary()

print('Compiling model')

opt = keras.optimizers.Adam(lr=1e-5, decay=1e-7 / 200)
metrics = ['accuracy']
loss = 'mean_squared_error'

autoencoder.compile(loss=loss,
              optimizer=opt,
              metrics=metrics)

print('Training model')

autoencoder.fit(train_faces, train_faces, epochs=50, shuffle=True)

model_json = autoencoder.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
autoencoder.save_weights("model.h5")