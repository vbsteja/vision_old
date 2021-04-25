#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 12:43:38 2021

@author: surya
"""

import random
import cv2
from matplotlib import pyplot as plt
import numpy as np


import albumentations as A



transforms = [A.PadIfNeeded(min_height=256, min_width=256, p=1),
              A.CenterCrop(p=1, height=256, width=256),
              A.HorizontalFlip(p=1),
              A.VerticalFlip(p=1),
              A.RandomRotate90(p=1),
              A.Transpose(p=1),
              A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
              A.GridDistortion(p=1),
              A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1),
              A.RandomSizedCrop(min_max_height=(50, 101), height=256, width=256, p=1),
              A.Compose([
                    A.VerticalFlip(p=0.5),              
                    A.RandomRotate90(p=0.5)]),
              A.CLAHE(p=0.8),
              A.RandomBrightnessContrast(p=0.8),    
              A.RandomGamma(p=0.8),
              A.Blur(p=0.8),
              A.ChannelShuffle(p=0.8),
              A.CoarseDropout(),
              A.ColorJitter(),
              A.Downscale(),
              A.Equalize(),
              A.MedianBlur(),
              A.MotionBlur(),
              A.RandomBrightness(),
              A.RandomFog(),
              A.RandomGridShuffle(),
              A.RandomRain(),
              A.RandomShadow(),
              A.RandomSnow(),
              A.RandomSunFlare()
              # A.RandomToneCurve()
            ]

def generate_aug_dataset(image, annotation):
    
    # annotation = json.load(open(os.path.join(self.images_directory, 
    #                                              "".join(image_filename.split(".")[:-1])+".json"),
    #                                 "r"))

    points = [a["points"] for a in annotation["shapes"] if a["shape_type"] == "polygon"]
    
    mask = np.zeros(image.shape[:2],dtype="uint8")
    for pt in points:
        pt = np.array(pt, np.int32).reshape(-1,1,2)
        cv2.fillPoly(mask, [pt], 1)
    
    aug_images = []
    aug_masks = []
    for t in transforms:
        #print(image.shape)
        transformed = t(image=image, mask=mask)
        aug_images.append(transformed["image"])
        aug_masks.append(transformed["mask"])
    return  aug_images, aug_masks