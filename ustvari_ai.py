from __future__ import print_function

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as itk
import tensorflow as tf

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import Callback

from os.path import exists, join
from sklearn.model_selection import train_test_split
# from amslib import load_mri_brain_data
from tqdm import tqdm
from PIL import Image as im

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)

# physical_devices = tf.config.experimental.list_physical_devices(‘GPU’)
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

seed = 42
random.seed = seed
np.random.seed = seed

CSF, GM, WM, LESIONS = 1, 2, 3, 10
TEST_DATA_FRACTION = 0.33
IMAGE_SIZE = (64, 64)
MODALITIES = ('t1',) # ('t1','flair') ali ('flair',) ali ('t1',)
NUM_CLASSES = 2 # binarno razvrščanje (lahko tudi več kategorij oz. oznak)

slike_sub_array = dict()
maske_sub_array = dict()

cebele_data = []
cebele_data_slike = []
cebele_data_maske = []

from os.path import join

DATA_PATH = '/Users/tilenkocevar/Documents/FAKS/Robotika/1.letnik/2.semester/Robotski vid/projekt - cebele/maske'
pacient_no = 0

# poišči vse podmape
patient_paths = os.listdir(DATA_PATH)
print('path', patient_paths)

print('V mapi {:s} je {:d} podmap.'.format(DATA_PATH, len(patient_paths)))

for pacient_no in tqdm(range(len(patient_paths))):
    if not patient_paths[pacient_no].startswith('.'):
        patient_path = join(DATA_PATH, patient_paths[pacient_no])
        t1 = itk.ReadImage(join(patient_path, 'slika.png'))
        bmsk = itk.ReadImage(join(patient_path, 'maska.nrrd'))

        t5 = itk.GetArrayFromImage(t1)
        bmsk5 = np.squeeze(itk.GetArrayFromImage(bmsk))

        for i in range(10):
            slike_sub_array[i] = np.array(t5[i * 384:(i + 1) * 384, i * 226:(i + 1) * 226])
            maske_sub_array[i] = np.array(bmsk5[i * 384:(i + 1) * 384, i * 226:(i + 1) * 226])
            cebele_data.append([(slike_sub_array[i]), (maske_sub_array[i])])
            cebele_data_slike.append((slike_sub_array[i]))
            cebele_data_maske.append((maske_sub_array[i]))


# X, Y_bmsk, Y_seg = load_mri_brain_data(output_size=IMAGE_SIZE, modalities=MODALITIES)
X_train, X_test, y_train, y_test = train_test_split(cebele_data_slike, cebele_data_maske , test_size=TEST_DATA_FRACTION)

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
y_train = y_train.reshape((67, 384, 226, -1))
y_test = y_test.reshape((33, 384, 226, -1))

X_train_2= X_train[:,:,:,0]

print(X_train_2.shape)

print('Velikost učne zbirke slik: {}'.format(X_train.shape))
print('Velikost testne zbirke slik: {}'.format(X_test.shape))
print('Velikost Y: {}'.format(y_train.shape))
print('Velikost Y test: {}'.format(y_test.shape))

# določimo mejno velikost lezij
LESION_SIZE_THRESHOLD = 30

# ANALIZA UČNEGA SETA PODATKOV
# izračunaj velikost maske lezij za vsako sliko
lesion_voxels = np.sum(np.sum(np.squeeze(y_train==LESIONS),axis=-1),axis=-1)

large_lesions = np.count_nonzero(lesion_voxels>LESION_SIZE_THRESHOLD)
small_lesions = np.count_nonzero(lesion_voxels<=LESION_SIZE_THRESHOLD)
print('Število slik z veliko prostornino lezij: {:d}'.format(large_lesions))
print('Število slik z malo prostornino lezij: {:d}'.format(small_lesions))

# pretvori vektor oznak razreda v binarno matriko oznak tipa 1-k
Y_train = keras.utils.to_categorical((lesion_voxels<=LESION_SIZE_THRESHOLD).astype('int'))

# ANALIZA TESTNEGA SETA PODATKOV
# izračunaj velikost maske lezij za vsako sliko
lesion_voxels = np.sum(np.sum(np.squeeze(y_test==LESIONS),axis=-1),axis=-1)

large_lesions = np.count_nonzero(lesion_voxels>LESION_SIZE_THRESHOLD)
small_lesions = np.count_nonzero(lesion_voxels<=LESION_SIZE_THRESHOLD)
print('Število slik z veliko prostornino lezij: {:d}'.format(large_lesions))
print('Število slik z malo prostornino lezij: {:d}'.format(small_lesions))

# pretvori vektor oznak razreda v binarno matriko oznak tipa 1-k
Y_test = keras.utils.to_categorical((lesion_voxels<=LESION_SIZE_THRESHOLD).astype('int'))
print(Y_test.shape)

### BEGIN SOLUTION
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=X_train.shape[1:]))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.75))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(NUM_CLASSES, activation='softmax'))

# povzetek strukture modela in števila parametrov
model.summary()
### END SOLUTION

BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=LEARNING_RATE), # Adadelta, RMSprop, SGD,...
              metrics=['accuracy'])

# pripravi izpis kriterijskih funkcij za Tensorboard
run_count = 0
while exists('./graphs/' + str(run_count)):
    run_count += 1

tbCallBack = keras.callbacks.TensorBoard(
    log_dir='./graphs/' + str(run_count),
    histogram_freq=0,
    write_graph=True,
    write_images=True)

class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

# zaženi učenje modela
model.fit(X_train, Y_train,
          batch_size=BATCH_SIZE,
          epochs=NUM_EPOCHS,
          verbose=1,
          validation_data=(X_test, Y_test),
          callbacks=[tbCallBack, TestCallback((X_test, Y_test))])

# shrani model v lokalno mapo
model.save_weights(join('models','lesion-classification-modalities[{}].h5'.format('+'.join(MODALITIES))))
print('Model je shranjen na disk!')