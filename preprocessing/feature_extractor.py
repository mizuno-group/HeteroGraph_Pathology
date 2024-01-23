# -*- coding: utf-8 -*-
"""
Created on 2024-01-19 (Fri) 17:34:41

Extract features from images

References
- https://github.com/BiomedSciAI/histocartography

@author: I.Azuma
"""
# %%
import importlib
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np

import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Dataset

# %%
class ImageToPatchDataset(Dataset):
    """Helper class to transform an image as a set of patched wrapped in a pytorch dataset"""

    def __init__(
        self,
        image: np.ndarray,
    ) -> None:
        """Create a dataset for a given image and extracted instance maps with desired patches.
           Patches have shape of (3, 256, 256) as defined by HoverNet model.

        Args:
            image (np.ndarray): RGB input image
        """
        self.image = image
        self.dataset_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.im_h = image.shape[0]
        self.im_w = image.shape[1]
        self.all_patches, self.coords = extract_patches_from_image(
            image, self.im_h, self.im_w
        )
        self.nr_patches = len(self.all_patches)
        self.max_y_coord = max([coord[-2] for coord in self.coords])
        self.max_x_coord = max([coord[-1] for coord in self.coords])

    def __getitem__(self, index: int) -> Tuple[int, torch.Tensor]:
        """Loads an image for a given instance maps index

        Args:
            index (int): patch index

        Returns:
            Tuple[int, torch.Tensor]: index, image as tensor
        """
        patch = self.all_patches[index]
        coord = self.coords[index]
        transformed_image = self.dataset_transform(Image.fromarray(patch))
        return coord, transformed_image

    def __len__(self) -> int:
        """Returns the length of the dataset

        Returns:
            int: Length of the dataset
        """
        return self.nr_patches

def extract_patches_from_image(image, im_h, im_w):
    STEP_SIZE = [164, 164]
    x, last_h, last_w = pad_image(image, im_h, im_w)
    sub_patches = []
    coords = []
    # generating subpatches from original
    for row in range(0, last_h, STEP_SIZE[0]):
        for col in range(0, last_w, STEP_SIZE[1]):
            win = x[row:row + WIN_SIZE[0],
                    col:col + WIN_SIZE[1]]
            sub_patches.append(win)
            # left, bottom, right, top
            coords.append([col, row, col + STEP_SIZE[0], row + STEP_SIZE[1]])
    return sub_patches, coords

def get_last_steps(length, msk_size, step_size):
    nr_step = math.ceil((length - msk_size) / step_size)
    last_step = (nr_step + 1) * step_size
    return int(last_step)

def pad_image(image, im_h, im_w):
    MASK_SIZE = [164, 164]
    WIN_SIZE = [256, 256]
    
    last_h = get_last_steps(im_h, MASK_SIZE[0], STEP_SIZE[0])
    last_w = get_last_steps(im_w, MASK_SIZE[1], STEP_SIZE[1])
    diff_h = WIN_SIZE[0] - STEP_SIZE[0]
    padt = diff_h // 2
    padb = last_h + WIN_SIZE[0] - im_h
    diff_w = WIN_SIZE[1] - STEP_SIZE[1]
    padl = diff_w // 2
    padr = last_w + WIN_SIZE[1] - im_w
    image = np.pad(image, ((padt, padb), (padl, padr), (0, 0)), 'reflect')
    return image, last_h, last_w

class LoadModel():
    def __init__(self,architecture='resnet50'):
        self.architecture = architecture
        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if cuda else "cpu")
    
    def dynamic_import_from(self, source_file: str, class_name: str) -> Any:
        """Do a from source_file import class_name dynamically

        Args:
            source_file (str): Where to import from
            class_name (str): What to import

        Returns:
            Any: The class to be imported
        """
        module = importlib.import_module(source_file)
        return getattr(module, class_name)
    
    def _remove_layers(self, model: nn.Module, extraction_layer: Optional[str] = None) -> nn.Module:
        """
        Returns the model without the unused layers to get embeddings.

        Args:
            model (nn.Module): Classification model.

        Returns:
            nn.Module: Embedding model.
        """
        if hasattr(model, "model"):
            model = model.model
            if extraction_layer is not None:
                model = _remove_modules(model, extraction_layer)
        if isinstance(model, torchvision.models.resnet.ResNet):
            if extraction_layer is None:
                # remove classifier
                model.fc = nn.Sequential()
            else:
                # remove all layers after the extraction layer
                model = _remove_modules(model, extraction_layer)
        else:
            # remove classifier
            model.classifier = nn.Sequential()
            if extraction_layer is not None:
                # remove average pooling layer if necessary
                if hasattr(model, 'avgpool'):
                    model.avgpool = nn.Sequential()
                # remove all layers in the feature extractor after the extraction layer
                model.features = _remove_modules(model.features, extraction_layer)
        return model
    
    def load_model(self):
        model_class = self.dynamic_import_from("torchvision.models", self.architecture)
        model = model_class(pretrained=True)
        model = model.to(self.device)
        model = self._remove_layers(model, extraction_layer=None)
        model.eval()
        self.model = model

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