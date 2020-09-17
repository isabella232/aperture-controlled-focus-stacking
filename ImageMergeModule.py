#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 17:11:32 2020

@author: zhuonanlin
"""
import os
import numpy as np
from pathlib import Path
import shutil
import time
import cv2 as cv
import bisect

from DCC_map import DCC_map
from OpticalFlow import OpticalFlow


class ImageMerge:

    def __init__(self, focal_stack, device_name, method):
        '''
        This is main image merging module. It contains methods:
            code : method
              1  : 'Laplacian',
              2  : 'Multi depth map',
              3  : 'Single depth map',
              4  : 'Single depth map + optical flow + reference fill', #defualt method
              5  : 'Single depth map + optical flow + best fill'

        Parameters
        ----------
        focal_stack : FocalStack object
            The focal stack object from FocalStack module, preparing all materials
            for image merging.

        Returns
        -------
        None.

        '''
        method_look_up_table = {
            1: 'Laplacian',
            2: 'Multi depth map',
            3: 'Single depth map',
            4: 'Single depth map + optical flow + reference fill', #defualt method
            5: 'Single depth map + optical flow + best fill'}

        self.method = method
        self.focal_stack = focal_stack
        self.output_dir = output_dir = './output/'
        temp_path = Path(output_dir)
        if temp_path.exists() and temp_path.is_dir():
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)
        self.device_name = device_name
        self.output_flags = {'output_fov': True, 'output_optical_flow': True}

        self.DCC_map = DCC_map[self.device_name]
        self.DAC2Pos = -0.75
        
        
        if self.output_flags['output_optical_flow']:
            os.mkdir(os.path.join(output_dir, 'optical_flow_aligned/'))

        tic_image_merge = time.perf_counter()
        if self.method == 1:
            self.merged_image, self.mask = self.image_merge_Laplacian()
        elif self.method == 2:
            self.merged_image, self.mask = self.image_merge_multi_depth()
        elif self.method == 3:
            self.merged_image, self.mask = self.image_merge_single_depth()
        elif self.method == 4:
            self.merged_image, self.mask = self.image_merge_single_depth_optical_flow_reference_fill()
        elif self.method == 5:
            self.merged_image, self.mask = self.image_merge_single_depth_optical_flow_best_fill()
        elif self.method == 6:
            self.merged_image, self.mask = self.image_merge_Laplacian_optical_flow_2()

        os.mkdir(os.path.join(output_dir, 'merged/'))
        cv.imwrite(
            os.path.join(
                output_dir,
                'merged/') +
            'merged.jpg',
            self.merged_image)
        cv.imwrite(
            os.path.join(
                output_dir,
                'merged/') +
                'mask.jpg',
                self.mask)


    def image_merge_Laplacian(self):
        '''
        class method merging image using local sharpest pixel

        Laplacian + Guassian blur + pick sharpest pixel


        Returns
        -------
        image_result : np.darray
            merged result

        '''
        laplacian_kernel_size = 13
        gaussian_kernel_size = 13
        laplacian_images = []
        gradient_angles = []
        images = []

        H, W, C = next(iter(self.focal_stack.focal_stack_images.values()))[
            'image'].shape
        image_result = np.empty((H, W, C))

        for p in sorted(self.focal_stack.focal_stack_images.keys()):
            img = self.focal_stack.focal_stack_images[p]
            img_laplacian = cv.Laplacian(
                cv.cvtColor(
                    img['image'],
                    cv.COLOR_BGR2GRAY),
                cv.CV_64F,
                ksize=laplacian_kernel_size)

            img_gaussian_blur = cv.GaussianBlur(
                img_laplacian, (gaussian_kernel_size, gaussian_kernel_size), sigmaX=0)
            laplacian_images.append(img_gaussian_blur)
            images.append(img['image'])
        laplacian_images = np.array(laplacian_images)

        # use the largest value in the focal stack
        mask = np.argmax(np.abs(laplacian_images), axis=0)
        image_result = mask[np.newaxis, :, :,
                            np.newaxis].choose(images).squeeze(axis=0)
        
        mask = mask / len(self.focal_stack.focal_stack_images)  * 255.
        return image_result, mask

    def image_merge_multi_depth(self):
        '''
        class method merging image using multiple depth map method

        Stack the depth maps and pick the pixel from the frame with smallest
        absolute disparity value.


        Returns
        -------
        image_result : np.array
            meraged image.
        mask : np.array
            selection mask indicates which frame the each pixel is chosen from.

        '''
        H, W, C = next(iter(self.focal_stack.focal_stack_images.values()))[
            'image'].shape
        image_result = np.empty((H, W, C))

        depth_map_images = []
        images = []

        for p, img in self.focal_stack.depth_maps.items():
            # depth_map = cv.resize(img['depth_map'], (W, H), interpolation=cv.INTER_NEAREST)
            depth_map = img['depth_map']
            # print(self.focal_stack.focal_stack_images[p]['rotation_flag'])
            if self.focal_stack.focal_stack_images[p]['rotation_flag'] is not None:
                depth_map = cv.rotate(
                    depth_map, self.focal_stack.focal_stack_images[p]['rotation_flag'])

            if self.focal_stack.focal_stack_images[p]['rotation_flag'] in (
                    cv.ROTATE_90_CLOCKWISE, cv.ROTATE_90_COUNTERCLOCKWISE):
                depth_map = cv.resize(depth_map, (W, H))
            else:
                depth_map = cv.resize(depth_map, (W, H))

            depth_map_images.append(depth_map)
            images.append(self.focal_stack.focal_stack_images[p]['image'])
            # print(depth_map.shape, self.focal_stack.focal_stack_images[p]['image'].shape)

        depth_map_images = np.array(depth_map_images)

        mask = np.argmin(np.abs(depth_map_images), axis=0)
        image_mask = (mask) / \
            (len(sorted(self.focal_stack.depth_maps.keys()))) * 255.

        image_result = mask[np.newaxis, :, :,
                            np.newaxis].choose(images).squeeze(axis=0)
        return image_result, image_mask

    def image_merge_single_depth(self):
        '''
        class method merging image using single depth map method

        Calculate the targeting frame lens position using the disparity value in the
        reference depth map.

        p_target = p_ref + DCC_map * disparity * DAC2Pos


        Returns
        -------
        image_result : np.array
            meraged image.
        mask : np.array
            selection mask indicates which frame the each pixel is chosen from.
        '''
        positions = sorted(self.focal_stack.depth_maps.keys())
        # depth_map_position_used = sorted(self.focal_stack.depth_maps.keys())[len(self.focal_stack.depth_maps.keys()) // 2]
        depth_map_position_used = positions[np.argmin(
            np.abs(np.array(positions) - self.focal_stack.ref_lens_position))]
        # print(depth_map_position_used)
        depth_map_used = self.focal_stack.depth_maps[depth_map_position_used]['depth_map']
        H, W, C = next(iter(self.focal_stack.focal_stack_images.values()))[
            'image'].shape
        temp_DDC_map = cv.resize(
            self.DCC_map, (W, H), interpolation=cv.INTER_LINEAR)
        depth_map_ref = cv.resize(depth_map_used, (W, H))
        if self.focal_stack.focal_stack_images[depth_map_position_used]['rotation_flag'] is not None:
            depth_map_ref = cv.rotate(
                depth_map_ref,
                self.focal_stack.focal_stack_images[depth_map_position_used]['rotation_flag'])
        delta_p = depth_map_ref * (temp_DDC_map) * \
            (-self.DAC2Pos) + depth_map_position_used

        image_result = np.empty((H, W, C))
        decision_mask = np.empty((H, W, C))
        for x in range(H):
            for y in range(W):
                new_p = delta_p[x, y]
                if new_p <= positions[0]:
                    choice = positions[0]
                elif new_p >= positions[-1]:
                    if new_p >= 310:
                        choice = min(306, positions[-1])
                    else:
                        choice = positions[-1]
                else:

                    idx = bisect.bisect(positions, new_p)
                    choice = positions[idx -
                                       1] if abs(positions[idx -
                                                           1] -
                                                 new_p) < abs(positions[idx] -
                                                              new_p) else positions[idx]

                image_result[x,y] = self.focal_stack.focal_stack_images[choice]['image'][x, y]
                delta_p[x, y] = positions.index(choice)
        image_mask = (delta_p) / (len(positions)) * 255.

        return image_result, image_mask

    def image_merge_single_depth_optical_flow_reference_fill(self):
        '''
        class method merging image using single depth map method + optical flow correction


        Calculate the targeting frame lens position using the disparity value in the
        reference depth map.

        p_target = p_ref + DCC_map * disparity * DAC2Pos,
        x_target, y_target = x + Sum(u), y + Sum(v)
        
        
        For pixel positions has no target pixel (e.g. occluded due to motion), fill in the pixel from
        the reference frame.

        Returns
        -------
        image_result : np.array
            meraged image.
        mask : np.array
            selection mask indicates which frame the each pixel is chosen from.


        '''
        tic_image_merge = time.perf_counter()
        self.optical_flow = OpticalFlow(self.focal_stack)
        positions = sorted(self.focal_stack.depth_maps.keys())
        depth_map_position_used = positions[np.argmin(
            np.abs(np.array(positions) - self.focal_stack.ref_lens_position))]
        depth_map_used = self.focal_stack.depth_maps[depth_map_position_used]['depth_map']
        H, W, C = next(iter(self.focal_stack.focal_stack_images.values()))[
            'image'].shape
        image_result = np.zeros((H, W, C)) - 1
        temp_DCC_map = cv.resize(
            self.DCC_map, (W, H), interpolation=cv.INTER_LINEAR)
        depth_map_ref = cv.resize(depth_map_used, (W, H))
        delta_p = depth_map_ref * (temp_DCC_map) * \
            (-self.DAC2Pos) + depth_map_position_used

        time_output_optical_flow = 0
        tic_optical_flow = time.perf_counter()
        for p in positions:
            if p != depth_map_position_used:
                image_optical_flow_aligned, flow_prediction = self.optical_flow.align_with_optical_flow(
                    self.focal_stack.focal_stack_images[depth_map_position_used]['image'], self.focal_stack.focal_stack_images[p]['image'], scale_factor=16)
                image_optical_flow_aligned = cv.medianBlur(
                    image_optical_flow_aligned.astype(np.uint8), ksize=5)
                self.focal_stack.focal_stack_images[p]['image'] = image_optical_flow_aligned

                if self.output_flags['output_optical_flow']:
                    tic_output_optical_flow = time.perf_counter()
                    cv.imwrite(
                        os.path.join(
                            self.output_dir,
                            'optical_flow_aligned/') +
                        self.focal_stack.focal_stack_images[p]['name'],
                        image_optical_flow_aligned)
                    toc_output_optical_flow = time.perf_counter()
                    time_output_optical_flow += toc_output_optical_flow - tic_output_optical_flow
        toc_optical_flow = time.perf_counter()

       

        for y in range(H):
            for x in range(W):
                # find targeting frame lens position
                new_p = delta_p[y, x]
                if new_p <= positions[0]:
                    choice = positions[0]
                elif new_p >= positions[-1]:
                    if new_p >= 310:
                        choice = min(306, positions[-1])
                    else:
                        choice = positions[-1]
                else:

                    idx = bisect.bisect(positions, new_p)
                    choice = positions[idx -
                                       1] if abs(positions[idx -
                                                           1] -
                                                 new_p) < abs(positions[idx] -
                                                              new_p) else positions[idx]
                # find targeting pixel position using optical flow
                image_result[y,
                             x] = self.focal_stack.focal_stack_images[choice]['image'][y,
                                                                                       x]
                delta_p[y, x] = positions.index(choice)
                
        image_result[image_result==0] = self.focal_stack.focal_stack_images[depth_map_position_used]['image'][image_result==0]
        image_mask = (delta_p) / (len(positions)) * 255.
        toc_image_merge = time.perf_counter()

    

        return image_result, image_mask

    def image_merge_single_depth_optical_flow_best_fill(self):
        '''
        class method merging image using single depth map method + optical flow correction


        Calculate the targeting frame lens position using the disparity value in the
        reference depth map.

        p_target = p_ref + DCC_map * disparity * DAC2Pos,
        x_target, y_target = x + Sum(u), y + Sum(v)
        
        For pixel positions has no target pixel (e.g. occluded due to motion), fill in the pixel from
        the best frame possible.
        
        Returns
        -------
        image_result : np.array
            meraged image.
        mask : np.array
            selection mask indicates which frame the each pixel is chosen from.


        '''
        tic_image_merge = time.perf_counter()
        self.optical_flow = OpticalFlow(self.focal_stack)
        positions = sorted(self.focal_stack.depth_maps.keys())

        depth_map_position_used = positions[np.argmin(
            np.abs(np.array(positions) - self.focal_stack.ref_lens_position))]
        depth_map_used = self.focal_stack.depth_maps[depth_map_position_used]['depth_map']
        H, W, C = next(iter(self.focal_stack.focal_stack_images.values()))[
            'image'].shape
        image_result = np.empty((H, W, C))
        temp_DCC_map = cv.resize(
            self.DCC_map, (W, H), interpolation=cv.INTER_LINEAR)
        depth_map_ref = cv.resize(depth_map_used, (W, H))
        if self.focal_stack.focal_stack_images[depth_map_position_used]['rotation_flag'] is not None:
            depth_map_ref = cv.rotate(
                depth_map_ref,
                self.focal_stack.focal_stack_images[depth_map_position_used]['rotation_flag'])
            image_result = cv.rotate(
                image_result,
                self.focal_stack.focal_stack_images[depth_map_position_used]['rotation_flag'])
            temp_DCC_map = cv.rotate(
                temp_DCC_map,
                self.focal_stack.focal_stack_images[depth_map_position_used]['rotation_flag'])
        delta_p = depth_map_ref * (temp_DCC_map) * \
            (-self.DAC2Pos) + depth_map_position_used

        time_output_optical_flow = 0
        tic_optical_flow = time.perf_counter()
        for p in positions:
            if p != depth_map_position_used:
                image_optical_flow_aligned, flow_prediction = self.optical_flow.align_with_optical_flow(
                    self.focal_stack.focal_stack_images[depth_map_position_used]['image'], self.focal_stack.focal_stack_images[p]['image'], scale_factor=16)
                image_optical_flow_aligned = cv.medianBlur(
                    image_optical_flow_aligned, ksize=5)
                self.focal_stack.focal_stack_images[p]['image'] = image_optical_flow_aligned
                if self.output_flags['output_optical_flow']:
                    tic_output_optical_flow = time.perf_counter()
                    cv.imwrite(
                        os.path.join(
                            self.output_dir,
                            'optical_flow_aligned/') +
                        self.focal_stack.focal_stack_images[p]['name'],
                        image_optical_flow_aligned)
                    toc_output_optical_flow = time.perf_counter()
                    time_output_optical_flow += toc_output_optical_flow - tic_output_optical_flow
        toc_optical_flow = time.perf_counter()
    
        for y in range(H):
            for x in range(W):
                # find targeting frame lens position
                new_p = delta_p[y, x]
                if new_p <= positions[0]:
                    choice = positions[0]
                elif new_p >= positions[-1]:
                    if new_p >= 310:
                        choice = min(306, positions[-1])
                    else:
                        choice = positions[-1]
                else:

                    idx = bisect.bisect(positions, new_p)
                    choice = positions[idx -
                                       1] if abs(positions[idx -
                                                           1] -
                                                 new_p) < abs(positions[idx] -
                                                              new_p) else positions[idx]
                # find targeting pixel position using optical flow
                image_result[y,
                             x] = self.focal_stack.focal_stack_images[choice]['image'][y,
                                                                                       x]
                delta_p[y, x] = positions.index(choice)
        delta_p = delta_p.astype(np.int)
        # fill in with next aviliable
        fill_area = np.where(image_result == 0)
        for y, x in zip(fill_area[0], fill_area[1]):
            has_pixel = False
            delta_index = 1
            while not has_pixel:
                position_index = delta_p[y, x] + delta_index
                if position_index == positions.index(depth_map_position_used) or (
                        position_index < 0 or position_index >= len(positions)):
                    break
                if 0 <= position_index < len(positions):
                    candidate_position = positions[position_index]
                    pixel = self.focal_stack.focal_stack_images[candidate_position]['image'][y, x]
                    if pixel.sum() > 1e-16:
                        has_pixel = True
                        image_result[y, x] = pixel
                        delta_p[y, x] = position_index
                if not has_pixel:
                    position_index = delta_p[y, x] - delta_index
                    if position_index == positions.index(
                            depth_map_position_used):
                        break
                    if 0 <= position_index < len(positions):
                        candidate_position = positions[position_index]
                        pixel = self.focal_stack.focal_stack_images[candidate_position]['image'][y, x]
                        if pixel.sum() > 1e-16:
                            has_pixel = True
                            image_result[y, x] = pixel
                            delta_p[y, x] = position_index
                delta_index += 1

            if not has_pixel:
                image_result[y, x] = self.focal_stack.focal_stack_images[depth_map_position_used]['image'][y, x]
                delta_p[y, x] = positions.index(depth_map_position_used)

        image_mask = (delta_p) / (len(positions)) * 255.
        toc_image_merge = time.perf_counter()

        return image_result, image_mask
