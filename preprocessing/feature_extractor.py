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

import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms

# %%
# TODO: load model

class LoadModel():
    def __Init__(self,architecture='resnet50'):
        self.architecture = architecture
    
    def dynamic_import_from(source_file: str, class_name: str) -> Any:
        """Do a from source_file import class_name dynamically

        Args:
            source_file (str): Where to import from
            class_name (str): What to import

        Returns:
            Any: The class to be imported
        """
        module = importlib.import_module(source_file)
        return getattr(module, class_name)
    
    def _remove_layers(model: nn.Module, extraction_layer: Optional[str] = None) -> nn.Module:
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
    
    def _get_num_features(model: nn.Module, patch_size: int) -> int:
        """
        Get the number of features of a given model.

        Args:
            model (nn.Module): A PyTorch model.
            patch_size (int): Desired size of patch.

        Returns:
            int: Number of output features.
        """
        dummy_patch = torch.zeros(1, 3, patch_size, patch_size).to(device)
        features = model(dummy_patch)
        return features.shape[-1]
    
    def load_model(self):
        model_class = dynamic_import_from("torchvision.models", architecture)
        model = model_class(pretrained=True)
        model = model.to(device)
        model = _remove_layers(model, extraction_layer=None)
        num_features = _get_num_features(model, patch_size)
        model.eval()

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