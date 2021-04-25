#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 16:30:51 2021

@author: surya
"""
"""Environment constants defined here"""


from pathlib import Path
import os

home = str(Path.home())


PROJECT_HOME = os.path.join(home, "Documents/vision_research/")
PROJECT_DATA_HOME = os.path.join(PROJECT_HOME, "dev/images/")
DATA_HOME = os.path.join(home, "Documents/Data/")
MODELS_HOME = os.path.join(home, "Documents/Models/")
LOGS_HOME = os.path.join(home, "Documents/Logs/")
os.environ["TORCH_HOME"] = MODELS_HOME