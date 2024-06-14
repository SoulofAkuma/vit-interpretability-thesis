import numpy as np
import timm
from src.datasets import MixedPredictiveImages, MostPredictiveImages, MostPredictiveImagesByBlock
import torch
import tqdm
from src.utils.transformation import transform_images, get_transforms
from src.utils.IndexDataset import IndexDataset
from timm.models.vision_transformer import VisionTransformer
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union
import sqlite3
import os
from itertools import product
from torch.utils.data import DataLoader

MOBILE_NET = None
VGG_16 = None
XCEPTION = None
EFFICIENT_NET = None
TINY_CONVNEXT = None
DENSENET = None

def transferability_score(dataset: Union[MixedPredictiveImages.MixedPredictiveImagesDataset,
                          MostPredictiveImages.MostPredictiveImagesDataset,
                          MostPredictiveImagesByBlock.MostPredictiveImagesByBlockDataset],
                          result_path: str,
                          batch_size: int=10,
                          show_progression: bool=True,
                          k: int=5,
                          device=None) -> Dict[Tuple[str, str, int, Tuple[str, str]], float]:
    """Calculate the trans

    Args:
        dataset (Union[MixedPredictiveImages.MixedPredictiveImagesDataset,MostPredictiveImages.MostPredictiveImagesDataset,MostPredictiveImagesByBlock.MostPredictiveImagesByBlockDataset]): _description_
        huggingface_model_descriptor (str): _description_
        result_path (str): _description_
        batch_size (int, optional): _description_. Defaults to 10.
        show_progression (bool, optional): _description_. Defaults to True.
        k (int, optional): _description_. Defaults to 5.
        device (_type_, optional): _description_. Defaults to None.

    Raises:
        NameError: If the file already exists

    Returns:
        _type_: _description_
    """    """"""
    global MOBILE_NET, VGG_16, XCEPTION, EFFICIENT_NET, TINY_CONVNEXT, DENSENET

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    MOBILE_NET = MOBILE_NET if MOBILE_NET is not None else timm.create_model('mobilenetv3_large_100', pretrained=True)
    VGG_16 = VGG_16 if VGG_16 is not None else timm.create_model('vgg16', pretrained=True)
    XCEPTION = XCEPTION if XCEPTION is not None else timm.create_model('xception', pretrained=True)
    EFFICIENT_NET = EFFICIENT_NET if EFFICIENT_NET is not None else timm.create_model('efficientnet_b0', pretrained=True)
    TINY_CONVNEXT = TINY_CONVNEXT if TINY_CONVNEXT is not None else timm.create_model('convnext_tiny',pretrained=True)
    DENSENET = DENSENET if DENSENET is not None else timm.create_model('densenet121',pretrained=True)

    models: Dict[str, nn.Module] = {
        'mobile_net': MOBILE_NET, 
        'vgg16': VGG_16, 
        'xception': XCEPTION,
        'efficient_net': EFFICIENT_NET,
        'tiny_convnext': TINY_CONVNEXT,
        'densenet': DENSENET,
    }

    for i in models.keys():
        models[i] = models[i].to(device)
        models[i].eval()

    loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=10)
    common_transform = get_transforms(MOBILE_NET)
    xception_transform = get_transforms(XCEPTION)

    transforms_cache = dataset.transforms
    dataset.transforms = {
        'common': common_transform,
        'xception': xception_transform
    }

    if os.path.exists(result_path):
        raise NameError('The result ' + result_path + ' already exists')
    
    connection = sqlite3.connect(result_path)
    cursor = connection.cursor()
    correct_logits = None
    
    predictions = [f"prediction{i}" for i in range(k)]

    if type(dataset) is MixedPredictiveImages.MixedPredictiveImagesDataset:
        connection.execute(f"""
            CREATE TABLE predictions (
                pred_model TEXT,
                gen_model TEXT,
                imagenet_id TEXT,
                num_idx INTEGER,
                name TEXT,
                iteration INTEGER,
                top TEXT,
                weighting TEXT,
                gen_type TEXT,
                prediction INTEGER{"," if len(predictions) > 1 else ""}
                {",".join([f"{prediction} INTEGER" for prediction in predictions])}
            )
        """)

        gen_types = [dataset.gen_type] if dataset.gen_type != 'all' else ['div', 'clear']

        correct_predictions = {
            k: 0
            for k in product(dataset.models, gen_types, dataset.iterations, dataset.weight_top_combs)
        }
        total_predictions = {
            k: 0
            for k in product(dataset.models, gen_types, dataset.iterations, dataset.weight_top_combs)
        }

    for i, items in enumerate(tqdm.tqdm(loader, disable=not show_progression)):
        
        items_len = len(items['imagenet_id'])

        common_tensor_imgs = items['img_common'].to(device)
        xception_tensor_imgs = items['img_xception'].to(device)
        
        for model_name, model in models.items():

            _, predicted_logits = model(xception_tensor_imgs if model_name == 'xception'
                                        else common_tensor_imgs).topk(k+1, dim=1)

            if type(dataset) is MixedPredictiveImages.MixedPredictiveImagesDataset:
                rows = []
                for ii in range(items_len):
                    key = (items['model'][ii], items['gen_type'][ii], items['iteration'][ii].item(), 
                           (items['top'][ii], items['weighting_scheme'][ii]))
                    correct_predictions[key] += (
                        items['num_idx'][ii].item()==predicted_logits[ii,0].item())
                    total_predictions[key] += 1
                    rows.append((model_name, items['model'][ii], items['imagenet_id'][ii], 
                                 items['num_idx'][ii].item(), items['name'][ii], 
                                 items['iteration'][ii].item(), items['top'][ii], items['weighting_scheme'][ii], items['gen_type'][ii], 
                                 predicted_logits[ii, 0].item(), 
                                 *[predicted_logits[ii, iii].item() for iii in range(k)]))

                cursor.executemany(f"""
                        INSERT INTO predictions (pred_model, gen_model, imagenet_id, num_idx, name,
                            iteration, top, weighting, gen_type, prediction
                            {"," if k > 0 else ""}
                            {", ".join(predictions)}) VALUES 
                        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                         {"," if k > 0 else ""}
                         {", ".join(["?" for _ in range(k)])})
                    """, rows
                )

        connection.commit()
    
    cursor.close()
    connection.close()
    dataset.transforms = transforms_cache

    return {
        k: (correct_predictions[k] / total_predictions[k]) if total_predictions[k] > 0 else 'No Predictions'
        for k in product(dataset.models, gen_types, dataset.iterations, dataset.weight_top_combs)
    }

def prepare_plausibility(model: VisionTransformer, dataset: IndexDataset, mmap_path: str,
                         huggingface_model_descriptor: str, batch_size: int=10, device: Optional[str]=None):
    """Prepare the calculation of plausibility score calculation by storing the feature embeddings
    of all elements in the dataset on disk

    Args:
        model (VisionTransformer): The vision transformer to get the feature embedding from 
        dataset (IndexDataset): The dataset to run through the ViT
        mmap_path (str): The path to store the result under
        huggingface_model_descriptor (str): The huggingface descriptor for the transformation pipeline.
        batch_size (int, optional): The batch size to load data with. Defaults to 10.
        device (str, optional): The device to run the model on. Defaults to None.
    """

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.eval().to(device)

    loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=10)

    feature_mmap = np.memmap(mmap_path, dtype=float, mode='w+', shape=len(loader.dataset, model.embed_dim))

    for i, items in enumerate(tqdm.tqdm(loader)):
        
        tensor_imgs = transform_images([items[ii]['img'] for ii in range(len(items))], 
                                       huggingface_model_descriptor, device, concat=True)
        
