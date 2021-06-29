# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 00:05:33 2020

@author: Yu Zhe
"""
from mrcnn.config import Config


class CustomConfig(Config):
    """
    Derives from the base Config class and overrides some values.
    """
    # Configuration name
    NAME = "scratch"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Car Background + scratch
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100
    # Confidence level
    DETECTION_MIN_CONFIDENCE = 0.9
