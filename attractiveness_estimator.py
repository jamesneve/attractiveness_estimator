# One-hot encode the result - DONE
# Simplify down to two classes - DONE
# Look at how many there are in each class - DONE, set bin to 1/2 way
# Plot confusion matrix of the result
# Try treating it as a regression problem - DONE(?)
# Perhaps simplify the model
# Look at other examples of face classification - DONE(?)



from __future__ import absolute_import, division, print_function

from tensorflow import keras

import numpy as np
import pandas as pd

import os
import cv2

CSV = "/Users/james.neve/Data/Attractiveness/FaceSamples/likes.csv"
FACES_DIR = "/Users/james.neve/Data/Attractiveness/ExtractedFaces/"

df = pd.read_csv(CSV)
bins = [0, 60, float("inf")]
labels = [0, 1]
one_hot_labels = np.array([keras.utils.to_categorical(k, num_classes=len(labels)) for k in labels])
df['binned_likes'] = pd.cut(df['likes'], bins=bins, labels=labels, include_lowest=True)

print("Stats for each bin:")
print(df.groupby(['binned_likes']).agg(['count']))


files = os.listdir(FACES_DIR)

print('Assigning training labels')

train_images = np.empty([len(files), 3, 500, 500])
# train_labels = np.empty([len(files), len(labels)])
train_labels = np.empty([len(files)])
for i, face_file in enumerate(files):

    if face_file[-4:] != ".jpg":
        continue

    image_id = int(face_file.split("_")[0])
    row = df.loc[df['image_id'] == image_id]
    if len(row) != 1:
        continue

    # train_labels[i] = one_hot_labels[row['binned_likes']]
    train_labels[i] = row['likes']

    path = "%s%s%s" % (FACES_DIR, "/", face_file)
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    # Normalise RBG
    for j in [0, 1, 2]:
        train_images[i, j] = img[:, :, j]

train_images = train_images / 255.0

print('Constructing network')

# Based on https://arxiv.org/pdf/1503.03832.pdf
model = keras.Sequential()
model.add(keras.layers.Conv2D(3, (7, 7), input_shape=(3, 500, 500), padding="same"))
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
metrics = []

model.compile(loss='mean_absolute_percentage_error',
              optimizer=opt,
              metrics=metrics)

print('Training model')

model.fit(train_images, train_labels, epochs=5)

# print('Testing model')

# test_loss, test_acc = model.evaluate(test_images, train_labels)
# print('Test accuracy:', test_acc)
