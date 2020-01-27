
# -- Jan 16-23
# One-hot encode the result - DONE, no big improvements
# Simplify down to two classes - DONE, no big improvements
# Look at how many there are in each class - DONE, set bin to 1/2 way
# Plot confusion matrix of the result - POSTPONED
# Try treating it as a regression problem - DONE, significant improvement
# Look at other examples of face classification - DONE, used network from FACENET, significant improvement

# -- Jan 23-30
# Generate test results
# Clean up data, filter by area, age(??), other factors? - DONE
# - Take very new users only - DONE
# - Only Tokyo - DONE
# - Normalise based on number of profile views? - IMPOSSIBLE :(
# Other ways to pull faces out of the data? Pre-trained face detector CNN
# - Instagram attractiveness detector
# Run network on original photos
# Look at main photo only - DONE
# Filter out users > 100 or so likes




from __future__ import absolute_import, division, print_function

from tensorflow import keras

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import os
import cv2

CSV = "/Users/james.neve/Data/Attractiveness/FaceSamplesBQ1/likes.csv"
FACES_DIR = "/Users/james.neve/Data/Attractiveness/FaceSamplesBQ1/"
#FACES_DIR = "/Users/james.neve/Data/Attractiveness/FaceSamplesM/"

df = pd.read_csv(CSV)

# bins = [0, 60, float("inf")]
# labels = [0, 1]
# one_hot_labels = np.array([keras.utils.to_categorical(k, num_classes=len(labels)) for k in labels])
# df['binned_likes'] = pd.cut(df['likes'], bins=bins, labels=labels, include_lowest=True)

# print("Stats for each bin:")
# print(df.groupby(['binned_likes']).agg(['count']))

files = os.listdir(FACES_DIR)

print('Assigning training labels')

train_images = np.empty([len(files), 3, 400, 400])
# train_labels = np.empty([len(files), len(labels)])
train_labels = np.empty([len(files)])
for i, face_file in enumerate(files):
    if face_file[-4:] != ".jpg":
        continue

    image_id = int(face_file.split("_")[0])
    row = df.loc[df['image'] == face_file]
    if len(row) != 1:
        continue

    # train_labels[i] = one_hot_labels[row['binned_likes']]
    train_labels[i] = row['likes']

    path = "%s%s" % (FACES_DIR, face_file)
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    if np.shape(img) != (400, 400, 3):
        continue

    # Normalise RBG
    for j in [0, 1, 2]:
        train_images[i, j] = img[:, :, j]

print('Normalising RGB')

train_images = train_images / 255.0

print('Constructing network')

# Based on https://arxiv.org/pdf/1503.03832.pdf
model = keras.Sequential()
model.add(keras.layers.Conv2D(3, (7, 7), input_shape=(3, 400, 400), padding="same"))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(3, 3), padding="same"))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv2D(64, (1, 1), padding="same"))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Conv2D(64, (3, 3), padding="same"))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(pool_size=(3, 3), padding="same"))

model.add(keras.layers.Conv2D(192, (1, 1), padding="same"))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Conv2D(192, (3, 3), padding="same"))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(3, 3), padding="same"))

model.add(keras.layers.Conv2D(384, (1, 1), padding="same"))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Conv2D(384, (3, 3), padding="same"))
model.add(keras.layers.Activation('relu'))

model.add(keras.layers.Conv2D(256, (1, 1), padding="same"))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Conv2D(256, (3, 3), padding="same"))
model.add(keras.layers.Activation('relu'))

model.add(keras.layers.Conv2D(256, (1, 1), padding="same"))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Conv2D(256, (3, 3), padding="same"))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(3, 3), padding="same"))


model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64))
model.add(keras.layers.Activation('relu'))
# model.add(keras.layers.Dense(2))
# model.add(keras.layers.Activation('softmax'))
model.add(keras.layers.Dense(1))
model.add(keras.layers.Activation('linear'))

print('Compiling model')

# opt = 'rmsprop'
opt = keras.optimizers.Adam(lr=1e-3, decay=1e-3 / 200)

# metrics = ['accuracy']
metrics = [keras.metrics.mae]

model.compile(loss='mean_squared_error',
              optimizer=opt,
              metrics=metrics)

print('Training model')

model.fit(train_images, train_labels, epochs=10, shuffle=True)

# print('Testing model')

# test_loss, test_acc = model.evaluate(test_images, train_labels)
# print('Test accuracy:', test_acc)
