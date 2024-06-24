import os
from re import S
from PIL import Image
import pandas as pd
from src.utils.IndexDataset import IndexDataset
from typing import Dict, Optional, Union, List
from src.utils.imagenet import MAPPING_FRAME, IMG_FRAME
import torchvision.transforms as tf

class ImageNetValDataset(IndexDataset):

    def __init__(self, dataset_path, 
                 transforms: Union[tf.Compose, Dict[str, tf.Compose], None]=None):
        super().__init__()
        self.data_path = os.path.join(dataset_path, 'val')
        self.locked_category = None
        self.locked_category_index = None
        self.nr_of_images_per_category = 50
        self.transforms = transforms

    def __len__(self):
        if self.locked_category is not None:
            return self.nr_of_images_per_category
        else:
            return len(IMG_FRAME.index)
            
    
    def lock_category(self, imagenet_id: Optional[str]=None):
        self.locked_category = imagenet_id
        self.locked_category_index = None if imagenet_id is None \
            else MAPPING_FRAME.loc[imagenet_id]['num_idx']
        
    def get_imagenet_classes(self) -> List[str]:
        return MAPPING_FRAME.index.tolist()

    def get_images_from_imgnet_id(self, imagenet_id: str) -> List:
        """Get all the items in the dataset from the given class

        Args:
            imagenet_id (str): The given imagenet class i.e. n0000000

        Returns:
            List: The list of items from the dataset
        """
        indices = IMG_FRAME.index[IMG_FRAME['imagenet_id'] == imagenet_id].tolist()
        return [self[index] for index in indices]    

    def __getitem__(self, index):
        item = super().__getitem__(index)
        if self.locked_category is not None:
            index = index * len(MAPPING_FRAME.index) + self.locked_category_index
        img_data = IMG_FRAME.iloc[index]
        cat_data = MAPPING_FRAME.loc[img_data['imagenet_id']]
        item['imagenet_id'] = img_data['imagenet_id']
        item['num_idx'] = cat_data['num_idx']
        item['name'] = cat_data['name']
        img_path = os.path.join(self.data_path, img_data['imagenet_id'], img_data['img_name'])
        item['img'] = Image.open(img_path) \
            .convert('RGB')
        item['path'] = img_path
        if type(self.transforms) is dict:
            for key, transforms in self.transforms.items():
                item[f'img_{key}'] = transforms(item['img'])
            del item['img']
        elif self.transforms is not None:
            item['img'] = self.transforms(item['img'])
        return item
    
class ImageNetTrainDataset(IndexDataset):

    def __init__(self, dataset_path, 
                 transforms: Union[tf.Compose, Dict[str, tf.Compose], None]=None):
        super().__init__()

        self.dataset_path = os.path.join(dataset_path, 'train')
        self.transforms = transforms
        self.nr_of_classes = len(MAPPING_FRAME.index)

        self.length = 0
        categories = [(f.name, f.path) for f in os.scandir(self.dataset_path) if f.is_dir()]
        self.categories = []
        self.category_lengths = {}
        self.start_inds = {}
        for category, category_path in categories:
            files = [(category, f.name) for f in os.scandir(category_path) if f.is_file()]
            self.length += len(files)
            self.category_lengths[category] = len(files)
            self.start_inds[category] = len(self.categories)
            self.categories.extend(files)

        self.locked_category = None

    def __len__(self):
        if self.locked_category is not None:
            return self.category_lengths[self.locked_category]
        else:
            return self.length
    
    def get_imagenet_classes(self) -> List[str]:
        return MAPPING_FRAME.index.tolist()
    
    def lock_category(self, imagenet_id: Optional[str]=None):
        self.locked_category = imagenet_id
    
    def __getitem__(self, index):
        item = super().__getitem__(index)
        if self.locked_category is not None:
            index += self.start_inds[self.locked_category]
        
        category, file_name = self.categories[index]
        category_data = MAPPING_FRAME.loc[category]
        
        item['num_idx'] = category_data['num_idx']
        item['imagenet_id'] = category
        img_path = os.path.join(self.dataset_path, category, file_name)
        item['name'] = category_data['name']
        item['path'] = img_path
        item['img'] = Image.open(img_path).convert('RGB')
        if type(self.transforms) is dict:
            for key, transforms in self.transforms.items():
                item[f'img_{key}'] = transforms(item['img'])
            del item['img']
        elif self.transforms is not None:
            item['img'] = self.transforms(item['img'])
        return item
