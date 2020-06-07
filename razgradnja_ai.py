import os
import sys
import random
import warnings

import numpy as np
import SimpleITK as itk
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf

from os.path import join
# from amslib import load_mri_brain_data

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

seed = 42
random.seed = seed
np.random.seed = seed


### Konstante in parametri

# nastavi željeno velikost slik
IMG_WIDTH = 256
IMG_HEIGHT = 384
MODALITIES = ('t1', 'flair') # 't1' and/or 'flair'
IMG_CHANNELS = len(MODALITIES)
# določi oznake
CSF, GM, WM, LESIONS = 1, 2, 3, 10
# določi delež testnih podatkov
TEST_DATA_FRACTION = 0.2


### 3.1 Naloži MRI podatke in loči med učne in testne

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

X_train, X_test, y_train, y_test = train_test_split(cebele_data_slike, cebele_data_maske , test_size=TEST_DATA_FRACTION)

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
y_train = y_train.reshape((80, 384, 226, -1))
y_test = y_test.reshape((20, 384, 226, -1))

## PADING
y_train = np.pad(y_train, [(0,0),(0,0),(0,30),(0,0)], mode='constant', constant_values=0)
y_test = np.pad(y_test, [(0,0),(0,0),(0,30),(0,0)], mode='constant', constant_values=0)

X_train = np.pad(X_train, [(0,0),(0,0),(0,30),(0,0)], mode='constant', constant_values=0)
X_test = np.pad(X_test, [(0,0),(0,0),(0,30),(0,0)], mode='constant', constant_values=0)


print('Velikost 4D polja za učenje: {}'.format(X_train.shape))
print('Velikost 4D polja za testiranje: {}'.format(X_test.shape))

print('Velikost Y polja za učenje: {}'.format(y_train.shape))
print('Velikost Y polja za testiranje: {}'.format(y_test.shape))


### Izris vzorca učnih podatkov

# Check if training data looks all right
ix = random.randint(0, X_train.shape[0])
_, _, num_modalities = X_train[ix].shape


print('index', ix)
titles = [m.upper() + ' slika' for m in MODALITIES] + ['Referenčna razgradnja']
f, ax = plt.subplots(1, num_modalities+1, sharex=True, sharey=True, figsize=(20, 5))
for i in range(num_modalities):
    ax[i].imshow(X_train[ix][:,:,i], cmap='gray')
    ax[i].set_title(titles[i])
    ax[i].axis('off')
ax[-1].imshow(np.squeeze(y_train[ix]))
ax[-1].set_title(titles[-1])

# plt.show()

### 3.2 Načrtovanje in učenje U-net modela

# vhodna plast
inputs = Input((IMG_HEIGHT, IMG_WIDTH, 3))
print(inputs)

### BEGIN SOLUTION
s = Lambda(lambda x: x) (inputs)

# levi del U
c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
c1 = Dropout(0.1) (c1)
c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
c2 = Dropout(0.1) (c2)
c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
c3 = Dropout(0.2) (c3)
c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
c4 = Dropout(0.2) (c4)
c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
c5 = Dropout(0.3) (c5)
c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

# desni del U
u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
c6 = Dropout(0.2) (c6)
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
print('U7', u7.shape)
print('C3', c3.shape)

c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
c7 = Dropout(0.2) (c7)
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

print('C7', c7.shape)


u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
print('U8', u8.shape)
print('C2', c2.shape)

u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
c8 = Dropout(0.1) (c8)
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
c9 = Dropout(0.1) (c9)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

# polno povezana plast z 1x1 konvolucijo
outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

# inicializacija modela
model = Model(inputs=[inputs], outputs=[outputs])
### END SOLUTION

# povzetek modela
model.summary()


### Definicija kriterijskih funkcij
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def iou_coef(y_true, y_pred, smooth=1):
    """
    IoU = (|X &amp; Y|)/ (|X or Y|)
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
    return (intersection + smooth) / ( union + smooth)

def iou_coef_loss(y_true, y_pred):
    return -iou_coef(y_true, y_pred)

def dice_coef(y_true, y_pred):
    """
    DSC = (2*|X &amp; Y|)/ (|X| + |Y|)
    """
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


### Nastavitve modela za učenje
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[dice_coef])

### Hiperparametri
BATCH_SIZE = 16
NUM_EPOCHS = 50

### Učenje modela

# zaženi učenje modela
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint(join('models', 'model-cebele-unet-seg-1.h5'), verbose=1, save_best_only=True)
results = model.fit(X_train, y_train, validation_split=0.1, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,
                    callbacks=[earlystopper, checkpointer])

