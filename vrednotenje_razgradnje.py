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



### 3.3 Vrednotenje razgradnje

# naloži model
model = load_model(join('models', 'model-cebele-unet-seg-1.h5'),
                   custom_objects={'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss})

# opravi razgradnjo na učni in testni zbirki
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_test = model.predict(X_test, verbose=1)

# dobljene vrednosti
preds_train_t = (preds_train > 0.3).astype(np.uint8)
preds_test_t = (preds_test > 0.3).astype(np.uint8)

# preveri kakovost razgradnje na učnih vzorcih (sanity check)
ix = random.randint(0, len(preds_train_t))
_, _, num_modalities = X_train[ix].shape

titles = [m.upper() + ' slika' for m in MODALITIES] + ['Referenčna razgradnja', 'Razgradnja U-net']
f, ax = plt.subplots(1, num_modalities+2, sharex=True, sharey=True, figsize=(20, 5))
for i in range(num_modalities):
    ax[i].imshow(X_train[ix][:,:,i], cmap='gray')
    ax[i].set_title(titles[i])
    ax[i].axis('off')
# prikaži referenčno razgradnjo
ax[-2].imshow(np.squeeze(y_train[ix]))
ax[-2].set_title(titles[-2])
ax[-2].axis('off')
# prikaži razgradnjo z U-net
ax[-1].imshow(np.squeeze(preds_train_t[ix]))
ax[-1].set_title(titles[-1])
ax[-1].axis('off')

plt.show()

# preveri kakovost razgradnje na naključno izbranih testnih vzorcih
ix = random.randint(0, len(preds_test_t))
_, _, num_modalities = X_test[ix].shape

titles = [m.upper() + ' slika' for m in MODALITIES] + ['Referenčna razgradnja', 'Razgradnja U-net']
f, ax = plt.subplots(1, num_modalities+2, sharex=True, sharey=True, figsize=(20, 5))
for i in range(num_modalities):
    ax[i].imshow(X_test[ix][:,:,i], cmap='gray')
    ax[i].set_title(titles[i])
    ax[i].axis('off')
# prikaži referenčno razgradnjo
ax[-2].imshow(np.squeeze(y_test[ix]))
ax[-2].set_title(titles[-2])
ax[-2].axis('off')
# prikaži razgradnjo z U-net
ax[-1].imshow(np.squeeze(preds_test_t[ix]))
ax[-1].set_title(titles[-1])
ax[-1].axis('off')

plt.show()

print('stevilo pixlov MASKA', np.count_nonzero(y_test[ix]))
print('stevilo pixlov Unet', np.count_nonzero(preds_test[ix]))

def dice(result, reference):
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    intersection = np.count_nonzero(result & reference)
    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)

    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0
    return dc

test_dice = []
for i in range(y_test.shape[0]):
    test_dice.append(dice(preds_test_t[i].flatten(), y_test[i].flatten()))

print('Povprečna vrednost Diceovega koeficienta na testni zbirki: ', np.mean(test_dice))