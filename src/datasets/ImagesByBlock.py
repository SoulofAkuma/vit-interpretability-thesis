from src.utils.IndexDataset import IndexDataset
import os

import pandas as pd
from PIL import Image
from typing import Dict, Literal, Optional, Set, Union, List, Tuple, TypedDict
from src.utils.imagenet import MAPPING_FRAME
from itertools import product
from torchvision.transforms.functional import pil_to_tensor
import torchvision.transforms as tf
import torch

iter_T = Literal[500]
types_T = Literal['clear', 'cls_token']
models_T = Literal[
    'vit_base_patch16_224',
    'vit_base_patch32_224',
    'vit_large_patch16_224',
    'vit_base_patch16_224_miil'
]

default_models = [
    'vit_base_patch16_224_miil',
    'vit_large_patch16_224',
    'vit_base_patch32_224',
    'vit_base_patch16_224',
]
default_blocks_by_model = {
    'vit_base_patch16_224_miil': [i for i in range(12)],
    'vit_large_patch16_224': [i for i in range(24)],
    'vit_base_patch32_224': [i for i in range(12)],
    'vit_base_patch16_224': [i for i in range(12)],
}

class ItemT(TypedDict):
    imagenet_id: str
    num_idx: int
    name: str
    iteration: iter_T
    model: models_T
    gen_type: types_T
    block: int
    img: Union[Image.Image,torch.Tensor]

class ImagesByBlockDataset(IndexDataset):

    def __init__(self, dataset_path: str, gen_types: Union[Literal['all'], types_T] = 'all',
                 iterations: Set[iter_T] = [500], models: Set[models_T] = default_models, 
                 block_by_model: Dict[str, List[int]] = default_blocks_by_model, 
                 transforms: Union[None, tf.Compose, Dict[str, tf.Compose]]=None):
        super().__init__()
        self.dataset_path = dataset_path
        self.gen_type  = gen_types
        self.transforms=transforms
        
        self.iterations = list(iterations)
        self.models = list(models)
        self.block_by_model = block_by_model
        self.gen_types = [gen_types] if gen_types != 'all' else ['clear', 'cls_token']
        
        self.model_block_combs: List[Tuple[str, int]] = [
            tuple(comb) for comb in [(model_name, block) 
                  for model_name in models for block in block_by_model[model_name]]
            ]

        self.type_factor = 2
        self.length = len(MAPPING_FRAME.index) * self.type_factor * len(iterations) * \
            len(self.model_block_combs)
        self.single_cat_length = self.type_factor * len(iterations) * len(models) \
            * len(self.model_block_combs)
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
        
        model_block_comb = self.model_block_combs[
            (index // len(MAPPING_FRAME.index)) % len(self.model_block_combs)]
        iteration = self.iterations[0]

        item['imagenet_id'] = cls.name
        item['num_idx'] = cls_ind
        item['name'] = cls['name']
        item['iteration'] = iteration
        item['model'] = model_block_comb[0]
        item['block'] = model_block_comb[1]
        
        gen_type_dir = self.gen_types[
            (index // len(MAPPING_FRAME.index) // len(self.model_block_combs)) % len(self.gen_types)]
        item['gen_type'] = gen_type_dir
        
        img_name = f'{cls.name}.png'
        img_path = os.path.join(self.dataset_path, model_block_comb[0], gen_type_dir, str(iteration), 
                                f'block_{model_block_comb[1]}', img_name)
        item['img'] = Image.open(img_path).convert('RGB')
        if type(self.transforms) is dict:
            for key, transforms in self.transforms.items():
                item[f'img_{key}'] = transforms(item['img'])
            del item['img']
        elif self.transforms is not None:
            item['img'] = self.transforms(item['img'])
        
        return item