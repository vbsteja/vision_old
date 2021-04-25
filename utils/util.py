#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 12:41:20 2021

@author: surya
"""
import torch
import torchvision
import kornia
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imutils

def imread(path, gray: bool = False):
    
    if gray:
        image = cv2.imread(path, 0)
    else:
        image = cv2.imread(path)
    return image

def imshow_kornia(input: torch.Tensor):
    """Plot Torch tensor using matplotlib,
    alternatively use imshow plt by sending a numpy array."""
    
    out: torch.Tensor = torchvision.utils.make_grid(input, nrow=2, padding=1)
    out_np: np.ndarray = kornia.tensor_to_image(out)
    plt.imshow(out_np)
    plt.axis('off')

def imshow_plt(input_image: np.ndarray, title = "Image"):
    "Plot Image using matplotlib"
    
    plt.figure(dpi=600)
    plt.imshow(input_image)
    plt.axis('off')
    
def imshow_plt_gray(input_image: np.ndarray, title = "Image"):
    "Plot Image using matplotlib"
    plt.figure(dpi=600)
    plt.imshow(input_image, cmap="gray")
    plt.axis('off')
    
def imshow_cv(input_image: np.ndarray,title = "Image", cvt_2_bgr: bool=False):
    "Plot Image using opencv"
    
    if cvt_2_bgr:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    cv2.imshow(title,input_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def resize_image_by_scale(image: np.ndarray, scale: int = 100):
    
    width = int(image.shape[1] * scale / 100)
    height = int(image.shape[0] * scale / 100)
    dim = (width, height)
    
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

def resize_image_by_wh(image: np.ndarray, width = None, height = None):
    
    if not width:
        width = image.shape[0]
    if not height:
        height = image.shape[1]
    dim = (width, height)
    
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

