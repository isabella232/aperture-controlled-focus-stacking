#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 03:21:44 2020

@author: zhuonanlin
"""


import argparse

from FocalStackModule import FocalStack
from ImageMergeModule import ImageMerge


parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, help='path to image folder')
parser.add_argument('--depth_path', type=str, help='path to depth map folder')
parser.add_argument('--ref_p', type=int, help='reference lens position')
parser.add_argument('--device_name', type=str, help='device name, corresponding DCC_map should be stored in DCC_map.py')
parser.add_argument('--merging_method', default=4, type=int, help='choice of image merging method, see ImageMargeModule.py for details')

opt = parser.parse_args()


image_path = opt.image_path
depth_path = opt.depth_path
device_name = opt.device_name
merging_method = opt.merging_method
reference_lens_position = opt.ref_p
# focal stack object, contains pre-prepared images and depth maps
focal_stack_object = FocalStack(image_data_path=image_path, depth_data_path=depth_path, ref_lens_position=reference_lens_position)
# image merge object, merge the image given the focal stack
image_merge_object = ImageMerge(focal_stack_object, device_name, method=merging_method)

