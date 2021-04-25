#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 12:20:09 2021

@author: surya
"""
import cv2
import numpy as np


def apply_binary_semantic_segmentation(image, predictions):
    """Make sure the image and predictions are of the same size"""
    
    mask = np.zeros_like(predictions)
    mask = mask.astype(np.uint8)
    # mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # mask_colored = mask_colored.astype(np.uint8)
    
    #pred = pred.astype('uint8')
    predictions [predictions > 0] = 255
    mask[100:,:] = predictions[100:,:]
    # pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
    
    # img = cv2.imread(os.path.join(test_images_directory, test_images_filenames[0]))
    # img = cv2.resize(image,(256,256))
    # img2 =img[100:,75:175].copy()
     
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    b,g,r = cv2.split(mask)
    mask = np.dstack((0*b,0*g,r))
    
    img = cv2.addWeighted(image,0.7,mask,0.3,0)
    # plt.figure(dpi=400)
    # plt.imshow(img[...,::-1])
    return img