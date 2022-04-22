# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 17:47:22 2022

@author: dec2g
"""
#https://stackoverflow.com/questions/63001988/how-to-remove-background-of-images-in-python"
# https://stackoverflow.com/questions/64491530/how-to-remove-the-background-from-a-picture-in-opencv-python
#https://becominghuman.ai/fix-opencv-imshow-not-working-51047a1f8dad

from skimage import io as skio
from skimage import filters

from scipy import ndimage as ndi
from skimage import morphology

import numpy as np

import matplotlib.pyplot as plt

import cv2


#%matplotlib inline
#Matplotlib parameters
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'
# plt.rcParams['figure.dpi'] = 200
# plt.close("all")

def load_img(image):
    pass

def preprocess_img(image):
    pass

def threshold_img(image):
    pass

# load image
img = cv2.imread(r"C:/Users/dec2g/GitHub/snap_vision_ads/dan_stuff/11059585_2.jpg")
hh, ww = img.shape[:2] # height and width
img.dtype # get image format (uint8 initially)

# Define the threshold on white
lower = np.array([200, 200, 200]) # Threshold
upper = np.array([255, 255, 255]) # White

# Create mask to only select black 
thresh = cv2.inRange(img, lower, upper)

# apply morphology
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# invert morp image
mask = 255 - morph

# apply mask to image
result = cv2.bitwise_and(img, img, mask=mask)

# save results
cv2.imshow('thresh', thresh)
cv2.imshow('morph', morph)
cv2.imshow('mask', mask)
cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
# # convert to graky
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # # threshold input image as mask
# mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)[1]

# # # negate mask
# mask = 255 - mask

# # apply morphology to remove isolated extraneous noise
# # use borderconstant of black since foreground touches the edges
# kernel = np.ones((3,3), np.uint8)
# mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
# mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# # anti-alias the mask -- blur then stretch
# # blur alpha channel
# mask = cv2.GaussianBlur(mask, (0,0), sigmaX=2, sigmaY=2, borderType = cv2.BORDER_DEFAULT)

# # linear stretch so that 127.5 goes to 0, but 255 stays 255
# mask = (2*(mask.astype(np.float32))-255.0).clip(0,255).astype(np.uint8)

# # put mask into alpha channel
# result = img.copy()
# result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
# result[:, :, 3] = mask

# # # save resulting masked image
# # cv2.imwrite('person_transp_bckgrnd.png', result)

# # # display result, though it won't show transparency
# cv2.imshow("INPUT", img)
# cv2.imshow("GRAY", gray)
# cv2.imshow("MASK", mask)
# cv2.imshow("RESULT", result)
# #cv2.waitKey(0)
# #cv2.destroyAllWindows()



#plt.imshow(img[:,:,::-1]) # array converts to correct colours for some reason https://stackoverflow.com/questions/50630825/matplotlib-imshow-distorting-colors
#plt.imshow(gray)
