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


# Encode
input = keras.layers.Input(shape=(FACE_WIDTH, FACE_HEIGHT, 3))

# x = keras.layers.Conv2D(3, (7, 7), activation="relu", padding="same")(input)
# x = keras.layers.MaxPooling2D(pool_size=(3, 3), padding="same")(x)

# x = keras.layers.Conv2D(16, (1, 1), activation="relu", padding="same")(x)
# x = keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
# x = keras.layers.MaxPooling2D(pool_size=(4, 4), padding="same")(x)
#
# x = keras.layers.Conv2D(32, (1, 1), activation="relu", padding="same")(x)
# x = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
# x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)
#
# x = keras.layers.Conv2D(64, (1, 1), activation="relu", padding="same")(x)
# x = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)

encoded = keras.layers.Dense(5)(input)



# Decode
x = keras.layers.Dense(5)(encoded)
# x = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
# x = keras.layers.Conv2D(64, (1, 1), activation="relu", padding="same")(x)
#
# x = keras.layers.UpSampling2D((2, 2))(x)
# x = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
# x = keras.layers.Conv2D(32, (1, 1), activation="relu", padding="same")(x)
#
# x = keras.layers.UpSampling2D((4, 4))(x)
# x = keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
# x = keras.layers.Conv2D(16, (1, 1), activation="relu", padding="same")(x)

# x = keras.layers.UpSampling2D((3, 3))(x)
decoded = keras.layers.Conv2D(3, (7, 7), activation="sigmoid", padding="same")(x)

# Autoencoder model
autoencoder = keras.Model(input, decoded)
input_autoencoder_shape = autoencoder.layers[0].input_shape[1:]
output_autoencoder_shape = autoencoder.layers[-1].output_shape[1:]

# Encoder model
encoder = keras.Model(input, encoded)
input_encoder_shape = encoder.layers[0].input_shape[1:]
output_encoder_shape = encoder.layers[-1].output_shape[1:]

# Decoder model
decoded_input = keras.Input(shape=output_encoder_shape)
# decoded_output = autoencoder.layers[-10](decoded_input)
# decoded_output = autoencoder.layers[-9](decoded_output)
# decoded_output = autoencoder.layers[-8](decoded_input)
# decoded_output = autoencoder.layers[-7](decoded_output)
# decoded_output = autoencoder.layers[-6](decoded_output)
# decoded_output = autoencoder.layers[-5](decoded_output)
# decoded_output = autoencoder.layers[-4](decoded_output)
# decoded_output = autoencoder.layers[-3](decoded_output)
decoded_output = autoencoder.layers[-2](decoded_input)
decoded_output = autoencoder.layers[-1](decoded_output)

decoder = keras.Model(decoded_input, decoded_output)
decoder_input_shape = decoder.layers[0].input_shape[1:]
decoder_output_shape = decoder.layers[-1].output_shape[1:]

print(autoencoder.summary())

print('Compiling model')

opt = keras.optimizers.Adam(lr=1e-5, decay=1e-7 / 200)
metrics = ['accuracy']
loss = 'binary_crossentropy'

autoencoder.compile(loss=loss,
              optimizer=opt,
              metrics=metrics)

print('Training model')
print(train_faces)
autoencoder.fit(train_faces, train_faces, epochs=50, shuffle=True)

autoencoder_json = autoencoder.to_json()
encoder_json = encoder.to_json()
decoder_json = decoder.to_json()
with open("saved_models/autoencoder.json", "w") as json_file:
    json_file.write(autoencoder_json)
with open("saved_models/encoder.json", "w") as json_file:
    json_file.write(encoder_json)
with open("saved_models/decoder.json", "w") as json_file:
    json_file.write(decoder_json)

autoencoder.save("saved_models/autoencoder.h5")
encoder.save("saved_models/encoder.h5")
decoder.save("saved_models/decoder.h5")