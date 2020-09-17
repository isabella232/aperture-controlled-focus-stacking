#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 18:47:39 2020

@author: zhuonanlin
"""
from __future__ import absolute_import, division, print_function
from copy import deepcopy
import os
import cv2 as cv
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tfoptflow/tfoptflow/'))
from model_pwcnet import ModelPWCNet, _DEFAULT_PWCNET_TEST_OPTIONS

class OpticalFlow:
    def __init__(self):
        '''
        Optical flow class to calculate optical flow prediction using PWC-net.
        
        PWC-net reference:
            https://github.com/philferriere/tfoptflow
            

        Returns
        -------
        None.

        '''
        self.pwc_net_weight_path = None
        if self.pwc_net_weight_path is None:
            sys.exit("Please specify the path to pwc_net wetight!")
        
    def build_pwc_net_model(self, H, W):
        '''
        Parameters
        ----------
        H, W : int
            image mering height/width.

        Returns
        -------
        None.

        '''
        gpu_devices = ['/device:CPU:0']
        controller = '/device:CPU:0'
        
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tfoptflow/', 'tfoptflow/')
        ckpt_path = self.pwc_net_weight_path

        # Configure the model for inference, starting with the default options
        nn_opts = deepcopy(_DEFAULT_PWCNET_TEST_OPTIONS)
        nn_opts['verbose'] = True
        nn_opts['ckpt_path'] = ckpt_path
        nn_opts['batch_size'] = 1
        nn_opts['gpu_devices'] = gpu_devices
        nn_opts['controller'] = controller

        # We're running the PWC-Net-large model in quarter-resolution mode
        # That is, with a 6 level pyramid, and upsampling of level 2 by 4 in
        # each dimension as the final flow prediction
        nn_opts['use_dense_cx'] = True
        nn_opts['use_res_cx'] = True
        nn_opts['pyr_lvls'] = 6
        nn_opts['flow_pred_lvl'] = 2

        # The size of the images in this dataset are not multiples of 64, while the model generates flows padded to multiples
        # of 64. Hence, we need to crop the predicted flows to their original
        # size
        nn_opts['adapt_info'] = (1, H, W, 2)

        # Instantiate the model in inference mode and display the model
        # configuration
        nn = ModelPWCNet(mode='test', options=nn_opts)
        nn.print_config()
        self.pwc_net_model = nn

    def align_with_optical_flow(
            self, image1, image2, scale_factor=16):
        '''
        Align image2 to image1 by optical flow

        Parameters
        ----------
        image1/image2 : np.array
            input image pairs.
        scale_factor : int, optional
            Downscale factors of input image size. The default is 16.

        Returns
        -------
        image_aligned : np.array
            optical aligned image
        
        flow : np.array
            optical flow prediction from image2 to image1

        '''
        # align image2 towards image1
        image1_scaled = cv.resize(
            image1,
            None,
            fx=1 / scale_factor,
            fy=1 / scale_factor,
            interpolation=cv.INTER_CUBIC)
        image2_scaled = cv.resize(
            image2,
            None,
            fx=1 / scale_factor,
            fy=1 / scale_factor,
            interpolation=cv.INTER_CUBIC)
        H, W, _ = image1.shape
        # Build a list of image pairs to process
        img_pairs = [(image2_scaled, image1_scaled)]

        if self.pwc_net_model is None:
            H_scaled, W_scaled, _ = image1_scaled.shape
            self.build_pwc_net_model(H_scaled, W_scaled)

        # Generate the predictions and display them
        pred_labels = self.pwc_net_model.predict_from_img_pairs(
            img_pairs, batch_size=1, verbose=True)

        flow_scaled = pred_labels[0]
        flow = cv.resize(
            flow_scaled *
            scale_factor,
            None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv.INTER_CUBIC)

        image_aligned = np.zeros_like(image1)

        x, y = np.meshgrid(np.arange(W), np.arange(H))

        newX = np.clip(np.round(x + flow[:, :, 0]), 0, W - 1).astype(np.int)
        newY = np.clip(np.round(y + flow[:, :, 1]), 0, H - 1).astype(np.int)
        # refine_newX, refine_newY = self.optical_flow_refine_with_BM(image1, image2, flow)
        image_aligned[newY, newX] = image2[y, x]

       
        return image_aligned, flow
