from types import MappingProxyType
from transformers import AutoImageProcessor
from transformers.models.vit.image_processing_vit import ViTImageProcessor
import transformers
from PIL import Image
from typing import Dict, List, Union
import torch
from torchvision import transforms as transforms
from timm.models.vision_transformer import VisionTransformer
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

__TRANSFORMATIONS: Dict[str, ViTImageProcessor] = {}

def transform_images(images: Union[Image.Image, List[Image.Image]], 
                     model: str, device=None, concat: bool=False) -> List[torch.Tensor]:
    """Perform the necessary image transformation for a given vision transformer
        on one or more images.

    Args:
        images (Union[Image.Image, List[Image.Image]]): One or more images
        model (str): The model descriptor from huggingface.co
        device (str, optional): A device to transfer the images to. Defaults to None.
        concat (bool, optional): True if the images are to be concatenated as a batch before being \
            returned. Only applies if images are a list. Defaults to False.

    Returns:
        List[torch.Tensor]: A list of image representations as tensor
    """

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model not in __TRANSFORMATIONS:
        __TRANSFORMATIONS[model] = AutoImageProcessor.from_pretrained(model)
    if type(images) is list:
        if concat:
            return __TRANSFORMATIONS[model](torch.concat(images, dim=0), 
                                            return_tensors='pt')['pixel_values'].to(device)
        else:
            return [__TRANSFORMATIONS[model](image, return_tensors='pt')['pixel_values'].to(device)
                    for image in images]
    else:
        return __TRANSFORMATIONS[model](images, return_tensors='pt')['pixel_values'].to(device)

def get_transforms(model: VisionTransformer) -> transforms.Compose:
    """Get the image transforms for a model

    Args:
        model (VisionTransformer): The model to get the transforms for

    Returns:
        transforms.Compose: The image transforms
    """

    config = resolve_data_config({}, model=model)
    return create_transform(**config)