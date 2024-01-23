# -*- coding: utf-8 -*-
"""
Created on 2024-01-23 (Tue) 21:57:56

@author: I.Azuma
"""
# %%
import importlib
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np

import torchvision.transforms as transforms

# %%
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

def patch_extractor(image, nuc_info, ext_method="centroid"):
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
            crop_img = cell_center_crop(image, center_x, center_y, patch_size = 72)
        elif ext_method == "bbox":
            crop_img = cell_bbox_crop(image, min_y, min_x, max_y, max_x)
        crop_img_list.append(crop_img)
        
    return crop_img_list, coords