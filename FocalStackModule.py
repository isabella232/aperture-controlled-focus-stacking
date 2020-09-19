#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 00:07:34 2020

@author: zhuonanlin
"""
import numpy as np
import os
import cv2 as cv


class FocalStack:
    def __init__(self, image_data_path, depth_data_path, ref_lens_position):
        '''
        Pseudo module to constuct focal stack. This module will output focal stack as input for
        ImageMerge module

        Parameters
        ----------
        image_data_path/depth_data_path : string
            path to image/depth
        ref_lens_position : int
            reference lens position for focal stack and merging

        Returns
        -------
        None.

        '''

        depth_data_path = depth_data_path
        image_data_path = image_data_path
        
        self.focal_stack_images = {int(p.split('_P')[1][:3]) : {'name' : p, 
                                                                'image': cv.imread(os.path.join(image_data_path, p))}
                                   for p in os.listdir(image_data_path) if p.endswith('.jpg')}
        self.depth_maps = {int(p.split('_P')[1][:3]) : {'name' : p, 
                                                                'depth_map': np.load(os.path.join(depth_data_path, p))}
                                   for p in os.listdir(depth_data_path) if p.endswith('.npy')}
        self.ref_lens_position = ref_lens_position
        
        
