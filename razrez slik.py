import os
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as itk
from PIL import Image as im
import cv2 as cv
import nrrd
#
# from amslib import resample_image

from os.path import join

DATA_PATH = '/Users/tilenkocevar/Documents/FAKS/Robotika/1.letnik/2.semester/Robotski vid/projekt - cebele/maske'
pacient_no = 0

# poišči vse podmape
patient_paths = os.listdir(DATA_PATH)
print('V mapi {:s} je {:d} podmap.'.format(DATA_PATH, len(patient_paths)))
##
# naloži podatke iz ene mape
patient_path = join(DATA_PATH, patient_paths[pacient_no])
# t1 = itk.ReadImage(join(patient_path, 'slika.png'))
t1 = itk.ReadImage('/Users/tilenkocevar/Documents/FAKS/Robotika/1.letnik/2.semester/Robotski vid/projekt - cebele/maske/1/slika.png')
# bmsk = itk.ReadImage(join(patient_path, 'maska.nrrd'))
bmsk = itk.ReadImage('/Users/tilenkocevar/Documents/FAKS/Robotika/1.letnik/2.semester/Robotski vid/projekt - cebele/maske/1/maska.nrrd')

print('Velikost posamezne slike je {}'.format(t1.GetSize()))

t2 = im.open('/Users/tilenkocevar/Documents/FAKS/Robotika/1.letnik/2.semester/Robotski vid/projekt - cebele/maske/1/slika.png')
t2 = np.asarray(t2)
#
# print('t2 shape:',t2.shape)
t3 = itk.GetArrayFromImage(t1)
print('t3 shape:', t3.shape)

bmsk3 = itk.GetArrayFromImage(bmsk)

print('Bmsk3 shape:', bmsk3.shape)
bmsk4 = np.squeeze(bmsk3)

readdata, header = nrrd.read('/Users/tilenkocevar/Documents/FAKS/Robotika/1.letnik/2.semester/Robotski vid/projekt - cebele/maske/1/maska.nrrd')
bmsk2 = np.asarray(readdata)
print('shape bmsk2:', bmsk2.shape)


def img2array(img):
    return np.squeeze(itk.GetArrayFromImage(img))


# prikaži podatke
f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(20, 5))
ax1.imshow(img2array(t1), cmap='gray')
ax1.set_title('Png slika')
ax1.axis('off')

ax2.imshow(img2array(bmsk), cmap='gray')
ax2.set_title('Maska slike')
ax2.axis('off')

# ax3.imshow(t2, cmap='gray')
# ax3.set_title('Maska slike')
# ax3.axis('off')
#
#
# ax4.imshow(t3, cmap='gray')
# ax4.set_title('bmsk2 slika')
# ax4.axis('off')

plt.show()