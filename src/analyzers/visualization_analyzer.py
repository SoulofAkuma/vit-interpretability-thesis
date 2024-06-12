import timm
from src.datasets import MixedPredictiveImages, MostPredictiveImages, MostPredictiveImagesByBlock
import torch
import tqdm
from src.utils.transformation import transform_images
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Union
import sqlite3
import os
from itertools import product

MOBILE_NET = None
VGG_16 = None
XCEPTION = None
EFFICIENT_NET = None
TINY_CONVNEXT = None
DENSENET = None

def transferability_score(dataset: Union[MixedPredictiveImages.MixedPredictiveImagesDataset,
                          MostPredictiveImages.MostPredictiveImagesDataset,
                          MostPredictiveImagesByBlock.MostPredictiveImagesByBlockDataset],
                          huggingface_model_descriptor: str,
                          result_path: str,
                          batch_size: int=10,
                          show_progression: bool=True,
                          device=None):

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

    loop = range(0, len(dataset), batch_size)
    if show_progression:
        loop = tqdm.tqdm(loop)

    if os.path.exists(result_path):
        raise NameError('The result ' + result_path + ' already exists')
    
    connection = sqlite3.connect(result_path)
    cursor = connection.cursor()
    correct_logits = None

    if type(dataset) is MixedPredictiveImages.MixedPredictiveImagesDataset:
        connection.execute("""
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
                prediction INTEGER
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

    for i in loop:
        range_size = min(batch_size, len(dataset) - i)
        items = [dataset[ii] for ii in range(i, i+range_size)]
        tensor_imgs = torch.concat(transform_images([items[ii]['img'] for ii in range(len(items))], 
                                                    huggingface_model_descriptor, device), dim=0)
        
        correct_logits = [items[ii]['num_idx'] for ii in range(len(items))]

        for model_name, model in models.items():

            predicted_logits = model(tensor_imgs).argmax(dim=1).tolist()

            if type(dataset) is MixedPredictiveImages.MixedPredictiveImagesDataset:
                rows = []
                for i, item in enumerate(items):
                    key = (item['model'], item['gen_type'], item['iteration'], 
                           (item['top'], item['weighting_scheme']))
                    correct_predictions[key] += correct_logits[i]==predicted_logits[0]
                    total_predictions[key] += 1
                    rows.append((model_name, item['model'], item['imagenet_id'], item['num_idx'],
                                 item['name'], item['iteration'], item['top'], item['weighting_scheme'],
                                 item['gen_type'], predicted_logits[i]))

                cursor.executemany("""
                        INSERT INTO predictions (pred_model, gen_model, imagenet_id, num_idx, name,
                            iteration, top, weighting, gen_type, prediction) VALUES 
                        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, rows
                )

        connection.commit()
    
    cursor.close()
    connection.close()

    return {
        k: (correct_predictions[k] / total_predictions[k]) if total_predictions[k] > 0 else 'No Predictions'
        for k in product(dataset.models, gen_types, dataset.iterations, dataset.weight_top_combs)
    }