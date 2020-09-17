#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import cv2 as cv
import sys
import collections


#Noting that for all the numbers in the DCC map, it should be divided by 4
#DCC_map is 8 by 6 array, the purpose of using a DCC map is to tell us the relationship
# between the disparity value and the logic position frame number

#DCC_map is a device dependent parameter, for new device, one should get these values
#first and put in this file


DCC_map = collections.defaultdict(np.array)


'''
Device: Pixel3-device1
-------------------DCC------------------
VersionNum: 0, MapWidth 8, MapHeight 6, Q factor 2
298, 251, 211, 199, 197, 203, 231, 280, 
286, 243, 210, 203, 200, 202, 221, 264, 
269, 234, 212, 206, 203, 204, 213, 242, 
267, 229, 211, 205, 205, 207, 215, 245, 
287, 234, 208, 200, 199, 200, 217, 255, 
297, 243, 207, 196, 195, 203, 225, 264, 
'''

DCC_map['Pixel3-device1'] = (np.array([298, 251, 211, 199, 197, 203, 231, 280, 
                                         286, 243, 210, 203, 200, 202, 221, 264, 
                                         269, 234, 212, 206, 203, 204, 213, 242, 
                                         267, 229, 211, 205, 205, 207, 215, 245, 
                                         287, 234, 208, 200, 199, 200, 217, 255, 
                                         297, 243, 207, 196, 195, 203, 225, 264]).reshape((8, 6)) / 4.).astype(np.float64)


'''
Device: Pixel3-device2
-------------------DCC------------------
VersionNum: 0, MapWidth 8, MapHeight 6, Q factor 2
331, 279, 237, 222, 224, 237, 283, 354, 
316, 267, 233, 222, 224, 235, 274, 341, 
306, 263, 236, 224, 226, 241, 275, 328, 
307, 264, 237, 224, 227, 239, 273, 325, 
316, 267, 233, 221, 224, 234, 272, 339, 
329, 281, 236, 220, 222, 236, 282, 349, 
'''
DCC_map['Pixel3-device2'] = (np.array([331, 279, 237, 222, 224, 237, 283, 354, 
                                         316, 267, 233, 222, 224, 235, 274, 341, 
                                         306, 263, 236, 224, 226, 241, 275, 328, 
                                         307, 264, 237, 224, 227, 239, 273, 325, 
                                         316, 267, 233, 221, 224, 234, 272, 339, 
                                         329, 281, 236, 220, 222, 236, 282, 349, ]).reshape((8, 6)) / 4.).astype(np.float64)

'''
Device: Pixel4-device1
-------------------DCC------------------
conversionCoefficientCount: 48, DCCMapWidth 8, DCCMapHeight 6, Q factor 2
190, 169, 143, 128, 129, 147, 172, 194, 
186, 166, 142, 128, 129, 145, 170, 190, 
186, 169, 146, 130, 132, 149, 172, 190, 
185, 168, 145, 130, 132, 152, 175, 190, 
184, 165, 141, 127, 129, 145, 168, 186, 
187, 167, 141, 126, 127, 143, 168, 188, 
'''
DCC_map['Pixel4-device1'] = (np.array([190, 169, 143, 128, 129, 147, 172, 194, 
                                              186, 166, 142, 128, 129, 145, 170, 190, 
                                              186, 169, 146, 130, 132, 149, 172, 190, 
                                              185, 168, 145, 130, 132, 152, 175, 190, 
                                              184, 165, 141, 127, 129, 145, 168, 186, 
                                              187, 167, 141, 126, 127, 143, 168, 188,]).reshape((8, 6)) / 4.).astype(np.float64)

