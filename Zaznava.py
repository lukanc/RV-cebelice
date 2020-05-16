#!/usr/bin/env


from tkinter import *
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image
import PIL.Image as im
import os
import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
import cv2 as cv



iImage = np.array(im.open('/Users/tilenkocevar/Documents/FAKS/Robotika/1.letnik/2.semester/Robotski vid/cebele/Spiclin2020/IMG_20190426_162927.jpg'))

fig =plt.figure(figsize=(5,4), dpi=100)
plt.imshow(iImage)
plt.show()


def canny(iImage, thr_type):
    if thr_type == 1:
        omega = 1 / 3
        imgG = cv.cvtColor(iImage, cv.COLOR_BGR2GRAY)
        m = np.mean(imgG)
        TL = max(0, (1 - omega) * m)
        TH = min(255, (1 + omega) * m)
        print('TL =', TL)
        print('TH =', TH)
        #         imgG = cv.cvtColor(iImage,cv.COLOR_BGR2GRAY)
        oEdge = cv.Canny(imgG, TL, TH)
        return oEdge

    if thr_type == 2:
        imgG = cv.cvtColor(iImage, cv.COLOR_BGR2GRAY)
        T0, thr = cv.threshold(imgG, 0, 255, cv.THRESH_OTSU)
        #         print(T0)
        TL = T0 / 2
        TH = T0
        print('TL =', TL)
        print('TH =', TH)
        oEdge = cv.Canny(imgG, TL, TH)
        return oEdge


oEdge = canny(iImage, 1)
plt.imshow(oEdge)
plt.show()

def CircleDetection(img):
#     img = cv.imread('data/slika4.jpg')
#     test = cv.imread('data/slika4.jpg')
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    imgHSV = cv.cvtColor(img,cv.COLOR_RGB2HSV)

    blur = cv.GaussianBlur(img,(3,3),1.1)
    imgG = cv.cvtColor(blur,cv.COLOR_RGB2GRAY)

    circles = cv.HoughCircles(imgG,cv.HOUGH_GRADIENT,1,20,
                                param1=120,param2=10,minRadius=30,maxRadius=50)

    # print(circles)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv.circle(img,(i[0],i[1]),2,(0,0,255),3)

    y = circles[:,:,0]
    x = circles[:,:,1]
#     print(x, y)

    circles = np.round(circles[0, :]).astype("int")
    no_of_circles = len(circles)
    print('Število cebel:',no_of_circles)
    plt.imshow(img)
    plt.show()

Image2 = cv.imread('/Users/tilenkocevar/Documents/FAKS/Robotika/1.letnik/2.semester/Robotski vid/cebele/Spiclin2020/IMG_20190426_162927.jpg')
CircleDetection(Image2)