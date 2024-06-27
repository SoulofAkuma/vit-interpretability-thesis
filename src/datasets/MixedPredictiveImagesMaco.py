from src.utils.IndexDataset import IndexDataset
import os

import pandas as pd
from PIL import Image
from typing import Dict, Literal, Optional, Set, Union, List, Tuple, TypedDict
from src.utils.imagenet import MAPPING_FRAME
from itertools import product
import torchvision.transforms as tf
import torch

iter_T = Literal[1000]
models_T = Literal[
    'vit_base_patch16_224',
    'vit_base_patch32_224',
    'vit_large_patch16_224',
    'vit_base_patch16_224_miil'
]
tops_T = Literal[
    '1',
    '1,2,3,4'
]
weighting_scheme_T = Literal['unweighted', 'softmax']

default_models = [
    'vit_base_patch16_224_miil',
    'vit_large_patch16_224',
    'vit_base_patch32_224',
    'vit_base_patch16_224',
]
default_tops = [
    '1,2,3,4',
    '1',
]

crop = tf.CenterCrop((640, 640)) # center crop for predictions as used by the MACO authors

class ItemT(TypedDict):
    imagenet_id: str
    num_idx: int
    name: str
    iteration: iter_T
    model: models_T
    top: tops_T
    weighting_scheme: weighting_scheme_T
    img: Union[Image.Image,torch.Tensor]

class MixedPredictiveImagesMacoDataset(IndexDataset):

    def __init__(self, dataset_path: str,
                 iterations: Set[iter_T] = [1000], models: Set[models_T] = default_models, 
                 tops: Set[tops_T] = default_tops, 
                 weighting_scheme: Union[Literal['all'], weighting_scheme_T] = 'all',
                 transforms: Union[None, tf.Compose, Dict[str, tf.Compose]]=None,
                 rgba: bool=False):
        super().__init__()
        self.dataset_path = dataset_path
        self.transforms=transforms
        
        self.iterations = list(iterations)
        self.models = list(models)
        self.rgba = rgba

        self.weight_top_combs: List[Tuple[tops_T, weighting_scheme_T]] = [
            tuple(comb) 
            for comb in product(list(filter(lambda x: x!='1', tops)), 
                                ['unweighted', 'softmax'] if weighting_scheme == 'all' else [weighting_scheme])
        ] + ([('1', 'unweighted')] if '1' in tops else [])
            
        self.type_factor = 1
        self.length = len(MAPPING_FRAME.index) * self.type_factor * len(iterations) * len(models) \
            * len(self.weight_top_combs)
        self.single_cat_length = self.type_factor * len(iterations) * len(models) \
            * len(self.weight_top_combs)
        self.locked_category = None

    def get_images_from_imgnet_id(self, imagenet_id: str):
        cls_ind = MAPPING_FRAME.loc[imagenet_id]['num_idx']
        return [self[index] for index in range(cls_ind, self.length, 1000)]

    def __len__(self):
        if self.locked_category is not None:
            return self.single_cat_length
        else:
            return self.length
    
    def lock_category(self, imagenet_id: Optional[str]=None):
        self.locked_category = imagenet_id

    def __getitem__(self, index) -> ItemT:
        item = super().__getitem__(index)
        if self.locked_category:
            index = index * 1000 + MAPPING_FRAME.loc[self.locked_category]['num_idx']

        cls_ind = index % len(MAPPING_FRAME.index)
        cls = MAPPING_FRAME.iloc[cls_ind]
        
        model = self.models[(index // len(MAPPING_FRAME.index)) % len(self.models)]
        
        weight_top_comb = self.weight_top_combs[
            (index // len(MAPPING_FRAME.index) // len(self.models)) % len(self.weight_top_combs)]
        batch_ind = (index // len(MAPPING_FRAME.index) // len(self.models) // 
                     len(self.weight_top_combs)) % self.type_factor
        iteration = self.iterations[
            index // len(MAPPING_FRAME.index) // len(self.models) // len(self.weight_top_combs) // self.type_factor]

        item['imagenet_id'] = cls.name
        item['num_idx'] = cls_ind
        item['name'] = cls['name']
        item['iteration'] = iteration
        item['model'] = model
        item['top'] = weight_top_comb[0]
        item['weighting_scheme'] = weight_top_comb[1]
        
        img_name = f'{cls.name}.png'
        img_path = os.path.join(self.dataset_path, model, str(iteration), 
                                f'top_{weight_top_comb[0]}', weight_top_comb[1], img_name)
        if self.rgba:
            item['img'] = Image.open(img_path).convert('RGBA')
        else:
            item['img'] = Image.open(img_path).convert('RGB')

        if type(self.transforms) is dict:
            for key, transforms in self.transforms.items():
                item[f'img_{key}'] = transforms(crop(item['img']))
            del item['img']
        elif self.transforms is not None:
            item['img'] = self.transforms(crop(item['img']))
        
        return item