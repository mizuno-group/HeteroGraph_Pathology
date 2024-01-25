# -*- coding: utf-8 -*-
"""
Created on 2024-01-23 (Tue) 21:57:56

@author: I.Azuma
"""
# %%
import importlib
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

import torchvision.transforms as transforms

# %% Cell
def cell_center_crop(image, center_x, center_y, patch_size = 72):
    center_x = int(round(center_x))
    center_y = int(round(center_y))

    top = center_x - (patch_size//2)
    left = center_y - (patch_size//2)
    ccrop = transforms.functional.crop(image,top,left,patch_size,patch_size)

    return ccrop

def cell_bbox_crop(image, min_y, min_x, max_y, max_x):
    """_summary_
    Args:
        min_y, min_x = cell_info['bbox'][0]
        max_y, max_x = cell_info['bbox'][1]

        img (PIL Image or Tensor) – Image to be cropped. (0,0) denotes the top left corner of the image.
    """
    ccrop = transforms.functional.crop(image,min_y,min_x,(max_y-min_y),(max_x-min_x))

    return ccrop

def cell_patch_extractor(image, nuc_info, ext_method="centroid", patch_size=72):
    ext_candi = ["centroid", "bbox"]

    crop_img_list = []
    coords = []
    for i, cell_id in enumerate(nuc_info):
        cell_info = nuc_info.get(cell_id)
        # Extrat centroid
        center_y, center_x = cell_info['centroid']
        center_x = int(round(center_x))
        center_y = int(round(center_y))

        # Extract bounding box
        min_y, min_x = cell_info['bbox'][0]
        max_y, max_x = cell_info['bbox'][1]
        coords.append([min_y, min_x, max_y, max_x])

        if ext_method == "centroid":
            crop_img = cell_center_crop(image, center_x, center_y, patch_size=patch_size)
        elif ext_method == "bbox":
            crop_img = cell_bbox_crop(image, min_y, min_x, max_y, max_x)
        crop_img_list.append(crop_img)
        
    return crop_img_list, coords

# %% Tissue
def sliding_window(superpixels, stepsize, windowsize):
    """Explore the best window place with sliding.

    Args:
        superpixels (np.array): e.g. Results obtained from ColorMergedSuperpixelExtractor().
        stepsize (int): Stepsize.
        windowsize (tuple): Windowsize. e.g. (144, 144)

    """
    for y in range(0, superpixels.shape[0], stepsize):
        for x in range(0, superpixels.shape[1], stepsize):
            yield ((x, y), superpixels[y:y + windowsize[1], x:x + windowsize[0]])

def tissue_upperleft_crop(image, ul_x, ul_y, patch_size=144):
    ul_x = int(round(ul_x))
    ul_y = int(round(ul_y))
    tcrop = transforms.functional.crop(image,ul_y,ul_x,patch_size,patch_size)

    return tcrop

def tissue_patch_extractor(image, superpixels, stepsize=10, windowsize=(144,144), mask_others=False):
    # Collect instance values
    features = []
    leftuppers = []
    windows = sliding_window(superpixels, stepsize, windowsize)
    for window in windows:
        leftuppers.append(window[0])
        features.append(window[-1])
    
    # Create instance counter map
    counter_array = np.zeros((len(features), superpixels.max()))
    for i,f in tqdm(enumerate(features)):
        uniqs, counts = np.unique(f, return_counts=True) # Faster than collections.Counter
        counter_array[i][uniqs-1] = counts
    
    # Extract the best window for each instance
    max_posi = counter_array.argmax(axis=0)

    # Best upperleft position
    ul_list = []
    for mp in max_posi:
        ul_list.append(leftuppers[mp])
    
    crop_img_list = []
    for idx, ul in enumerate(ul_list):
        if mask_others:
            tmp = np.array(image)
            mask = np.where(superpixels!=idx+1) # (x_array, y_array)
            tmp[mask[0],mask[1],:] = [0,0,0]
            input_image = transforms.functional.to_pil_image(tmp)
        else:
            input_image = image
        crop_img = tissue_upperleft_crop(input_image, ul[0], ul[1], patch_size=windowsize[0])
        crop_img_list.append(crop_img)

    return crop_img_list, ul_list
