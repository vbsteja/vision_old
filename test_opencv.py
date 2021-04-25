#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 10:16:41 2021

@author: surya
"""
#%%
import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
from vision.utils.imutils import *

from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize
from skimage import data
from skimage.util import invert
    
#%%
img1 = imread("vision/data/Chilli.jpg")

# img1 = cv.cvtColor(img1, cv.COLOR_BGR2HSV)
#show(img, "chilli")
#img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# show(img1, "CHilli in RGB")

imshow_plt(img1)

#%%
#Binarize image using skimage

thresh = threshold_otsu(img1)
binary = img1 > thresh
imshow_plt_gray(binary.astype(int))
#%%

print(img1.shape)

a = np.array([[[1,2,3],[1,2,3]],
              [[1,2,3],[1,2,3]]])
print(a.shape)
# dst = cv.addWeighted(img1,0.7,img2,0.3,0)

#%%
from PIL import Image

image = Image.open("data/Chilli.jpg")
image_pix = np.array(image.getdata())
print(image_pix.shape)