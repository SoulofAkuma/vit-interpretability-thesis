from src.utils.IndexDataset import IndexDataset
import os

import pandas as pd
from PIL import Image
from typing import Literal, Set, Union, List, Tuple, TypedDict
from src.utils.imagenet import MAPPING_FRAME
from itertools import product

iter_T = Literal[500]
types_T = Literal['div', 'clear']
models_T = Literal[
    'vit_base_patch16_224',
    'vit_base_patch32_224',
    'vit_large_patch16_224',
    'vit_base_patch16_224_miil'
]
tops_T = Literal[
    '1',
    '1,2',
    '1,2,3',
    '1,2,3,4'
]
weighting_scheme_T = Literal['unweighted', 'softmax']

default_models = {
    'vit_base_patch16_224',
    'vit_base_patch32_224',
    'vit_large_patch16_224',
    'vit_base_patch16_224_miil'
}
default_tops = {
    '1',
    '1,2',
    '1,2,3',
    '1,2,3,4'
}

class ItemT(TypedDict):
    imagenet_id: str
    num_idx: int
    name: str
    iteration: iter_T
    model: models_T
    top: tops_T
    weighting_scheme: weighting_scheme_T
    gen_type: types_T
    img: Image.Image

class MixedPredictiveImagesDataset(IndexDataset):

    def __init__(self, dataset_path: str, gen_types: Union[Literal['all'], types_T] = 'all',
                 iterations: Set[iter_T] = [500], models: Set[models_T] = default_models, 
                 tops: Set[tops_T] = default_tops, 
                 weighting_scheme: Union[Literal['all'], weighting_scheme_T] = 'all'):
        super().__init__()
        self.dataset_path = dataset_path
        self.gen_type  = gen_types
        
        self.iterations = list(iterations)
        self.models = list(models)
        
        self.weight_top_combs: List[Tuple[tops_T, weighting_scheme_T]] = [
            tuple(comb) 
            for comb in product(list(tops - {'1'}), 
                                ['unweighted', 'softmax'] if weighting_scheme == 'all' else [weighting_scheme])
        ] + ([('1', 'unweighted')] if '1' in tops else [])
            
        self.type_factor = {'all': 4, 'div': 3, 'clear': 1}[self.gen_type]
        self.length = len(MAPPING_FRAME.index) * self.type_factor * len(iterations) * len(models) \
            * len(self.weight_top_combs)

    def get_images_from_imgnet_id(self, imagenet_id: str):
        cls_ind = MAPPING_FRAME.loc[imagenet_id]['num_idx']
        return [self[index] for index in range(cls_ind, self.length, 1000)]

    def __len__(self):
        return self.length
    
    def __getitem__(self, index) -> ItemT:
        item = super().__getitem__(index)
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
        
        gen_type_dir = self.gen_type if self.gen_type != 'all' else ('div' if batch_ind < 3 else 'clear')
        if gen_type_dir == 'div': item['batch_ind'] = batch_ind
        item['gen_type'] = gen_type_dir
        
        img_name = f'{cls.name}{("_" + str(batch_ind)) if gen_type_dir == "div" else ""}.png'
        img_path = os.path.join(self.dataset_path, model, gen_type_dir, str(iteration), 
                                f'top_{weight_top_comb[0]}', weight_top_comb[1], img_name)
        item['img'] = Image.open(img_path).convert('RGB')
        return item