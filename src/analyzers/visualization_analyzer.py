import numpy as np
import timm
from src.datasets import MixedPredictiveImages, MostPredictiveImages, MostPredictiveImagesByBlock
import torch
from tqdm.auto import tqdm
from src.datasets import ImagesByBlock
from src.datasets.ImageNet import ImageNetTrainDataset, ImageNetValDataset
from src.datasets.ImagesByBlock import ImagesByBlockDataset
from src.utils.transformation import transform_images, get_transforms
from timm.models.vision_transformer import VisionTransformer
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union
import sqlite3
import os
from itertools import product
from torch.utils.data import DataLoader
from torchvision.models.feature_extraction import create_feature_extractor
from src.datasets.MixedPredictiveImagesMaco import MixedPredictiveImagesMacoDataset
from scipy import linalg
import json

MOBILE_NET = None
VGG_16 = None
XCEPTION = None
EFFICIENT_NET = None
TINY_CONVNEXT = None
DENSENET = None

gen_models = [
    'vit_base_patch16_224', 
    'vit_base_patch16_224_miil', 
    'vit_base_patch32_224', 
    'vit_large_patch16_224'
]

def transferability_score(dataset: Union[MixedPredictiveImages.MixedPredictiveImagesDataset,
                          MostPredictiveImages.MostPredictiveImagesDataset,
                          MostPredictiveImagesByBlock.MostPredictiveImagesByBlockDataset,
                          MixedPredictiveImagesMacoDataset,
                          ImagesByBlockDataset],
                          result_path: str,
                          batch_size: int=3,
                          show_progression: bool=True,
                          k: int=5, include_gen_models: bool=False,
                          device: Optional[str]=None
                          ) -> Dict[Tuple[str, str, int, Tuple[str, str]], float]:
    """Calculate the trans

    Args:
        dataset (Union[MixedPredictiveImages.MixedPredictiveImagesDataset,MostPredictiveImages.MostPredictiveImagesDataset,MostPredictiveImagesByBlock.MostPredictiveImagesByBlockDataset]): The dataset to compute the score for
        result_path (str): The path the result database will be saved under.
        batch_size (int, optional): The batch size used for computation. Defaults to 10.
        show_progression (bool, optional): Whether to show tqdm progression. Defaults to True.
        k (int, optional): The topk to store predictions for. Defaults to 5.
        include_gen_models (bool, optional): Whether to also calculate the scores for the generator model. Defaults to False.
        device (Optional[str], optional): The device to compute on. Defaults to None.

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

    common_transform = get_transforms(MOBILE_NET)
    xception_transform = get_transforms(XCEPTION)

    transforms_cache = dataset.transforms
    dataset.transforms = {
        'common': common_transform,
        'xception': xception_transform
    }

    if include_gen_models: 
        for generator_model in gen_models:
            models[generator_model] = timm.create_model(generator_model, pretrained=True)
        
        ds_transforms = dataset.transforms
        
        ds_transforms['vit_common'] = get_transforms(models['vit_base_patch16_224'])
        ds_transforms['vit_miil'] = get_transforms(models['vit_base_patch16_224_miil'])

    for i in models.keys():
        models[i] = models[i].to(device)
        models[i].eval()

    if os.path.exists(result_path):
        raise NameError('The result ' + result_path + ' already exists')
    
    sqlite3.register_adapter(np.int64, int)
    sqlite3.register_adapter(np.int32, int)

    connection = sqlite3.connect(result_path)
    cursor = connection.cursor()
    
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
    elif type(dataset) is MixedPredictiveImagesMacoDataset:
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
                prediction INTEGER{"," if len(predictions) > 1 else ""}
                {",".join([f"{prediction} INTEGER" for prediction in predictions])}
            )
        """)

        correct_predictions = {
            k: 0
            for k in product(dataset.models, dataset.iterations, dataset.weight_top_combs)
        }
        total_predictions = {
            k: 0
            for k in product(dataset.models, dataset.iterations, dataset.weight_top_combs)
        }
    elif type(dataset) is ImagesByBlockDataset:
        connection.execute(f"""
            CREATE TABLE predictions (
                pred_model TEXT,
                gen_model TEXT,
                imagenet_id TEXT,
                num_idx INTEGER,
                name TEXT,
                iteration INTEGER,
                block INTEGER,
                gen_type TEXT,
                prediction INTEGER{"," if len(predictions) > 1 else ""}
                {",".join([f"{prediction} INTEGER" for prediction in predictions])}
            )
        """)

        correct_predictions = {
            k: 0
            for k in product(dataset.model_block_combs, dataset.gen_types, dataset.iterations)
        }
        total_predictions = {
            k: 0
            for k in product(dataset.model_block_combs, dataset.gen_types, dataset.iterations)
        }
    else:
        print('Not implemented')
        return 'Not Implemented'

    # loader = DataLoader(dataset, batch_size, False, num_workers=1)

    # for i, items in enumerate(tqdm(loader, disable=not show_progression)):
        
    #     length = len(items['imagenet_id'])

    #     common_tensor_imgs = items['img_common'].to(device)
    #     xception_tensor_imgs = items['img_xception'].to(device)
        
    #     for model_name, model in models.items():

    #         _, predicted_logits = model(xception_tensor_imgs if model_name == 'xception'
    #                                     else common_tensor_imgs).topk(k+1, dim=1)

    #         if type(dataset) is MixedPredictiveImages.MixedPredictiveImagesDataset:
    #             rows = []
    #             for ii in range(length):
    #                 key = (items['model'][ii], items['gen_type'][ii], items['iteration'][ii].item(), 
    #                        (items['top'][ii], items['weighting_scheme'][ii]))
    #                 correct_predictions[key] += (
    #                     items['num_idx'][ii].item()==predicted_logits[ii,0].item())
    #                 total_predictions[key] += 1
    #                 rows.append((model_name, items['model'][ii], items['imagenet_id'][ii], 
    #                              items['num_idx'][ii].item(), items['name'][ii], 
    #                              items['iteration'][ii].item(), items['top'][ii], 
    #                              items['weighting_scheme'][ii],
    #                              items['gen_type'][ii], predicted_logits[ii, 0].item(), 
    #                              *[predicted_logits[ii, iii].item() for iii in range(k)]))

    #             cursor.executemany(f"""
    #                     INSERT INTO predictions (pred_model, gen_model, imagenet_id, num_idx, name,
    #                         iteration, top, weighting, gen_type, prediction
    #                         {"," if k > 0 else ""}
    #                         {", ".join(predictions)}) VALUES 
    #                     (?, ?, ?, ?, ?, ?, ?, ?, ?, ?
    #                      {"," if k > 0 else ""}
    #                      {", ".join(["?" for _ in range(k)])})
    #                 """, rows
    #             )
    #         elif type(dataset) is MixedPredictiveImagesMacoDataset:
    #             rows = []
    #             for ii in range(length):
    #                 key = (items['model'][ii], items['iteration'][ii].item(), 
    #                        (items['top'][ii], items['weighting_scheme'][ii]))
    #                 correct_predictions[key] += (
    #                     items['num_idx'][ii].item()==predicted_logits[ii,0].item())
    #                 total_predictions[key] += 1
    #                 rows.append((model_name, items['model'][ii], items['imagenet_id'][ii], 
    #                              items['num_idx'][ii].item(), items['name'][ii], 
    #                              items['iteration'][ii].item(), items['top'][ii], 
    #                              items['weighting_scheme'][ii], 
    #                              predicted_logits[ii, 0].item(), 
    #                              *[predicted_logits[ii, iii].item() for iii in range(k)]))

    #             cursor.executemany(f"""
    #                     INSERT INTO predictions (pred_model, gen_model, imagenet_id, num_idx, name,
    #                         iteration, top, weighting, prediction
    #                         {"," if k > 0 else ""}
    #                         {", ".join(predictions)}) VALUES 
    #                     (?, ?, ?, ?, ?, ?, ?, ?, ?
    #                      {"," if k > 0 else ""}
    #                      {", ".join(["?" for _ in range(k)])})
    #                 """, rows
    #             )

    #     connection.commit()
        # del items['img_common'], items['img_xception']
        # del common_tensor_imgs, xception_tensor_imgs, predicted_logits
        # torch.cuda.empty_cache()

    for i in tqdm(range(0, len(dataset), batch_size), disable=not show_progression):
        
        length = min(len(dataset)-i, batch_size)
        items = [dataset[ii+i] for ii in range(length)]

        common_tensor_imgs = torch.stack([items[ii]['img_common'] 
                                          for ii in range(length)]).to(device)
        xception_tensor_imgs = torch.stack([items[ii]['img_xception'] 
                                            for ii in range(length)]).to(device)
        vit_common_tensor_imgs = torch.stack([items[ii]['img_vit_common']
                                              for ii in range(length)]).to(device)
        vit_miil_tensor_imgs = torch.stack([items[ii]['img_vit_miil']
                                              for ii in range(length)]).to(device)
        
        for model_name, model in models.items():

            tensor_imgs = None
            if model_name.endswith('miil'):
                tensor_imgs = vit_miil_tensor_imgs
            elif model_name.startswith('vit'):
                tensor_imgs = vit_common_tensor_imgs
            elif model_name == 'xception':
                tensor_imgs = xception_tensor_imgs
            else:
                tensor_imgs = common_tensor_imgs

            _, predicted_logits = model(tensor_imgs).topk(k+1, dim=1)

            if type(dataset) is MixedPredictiveImages.MixedPredictiveImagesDataset:
                rows = []
                for ii in range(length):
                    key = (items[ii]['model'], items[ii]['gen_type'], items[ii]['iteration'], 
                           (items[ii]['top'], items[ii]['weighting_scheme']))
                    correct_predictions[key] += (
                        items[ii]['num_idx']==predicted_logits[ii,0].item())
                    total_predictions[key] += 1
                    rows.append((model_name, items[ii]['model'], items[ii]['imagenet_id'], 
                                 items[ii]['num_idx'], items[ii]['name'], 
                                 items[ii]['iteration'], items[ii]['top'], 
                                 items[ii]['weighting_scheme'],
                                 items[ii]['gen_type'], predicted_logits[ii, 0].item(), 
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
            elif type(dataset) is MixedPredictiveImagesMacoDataset:
                rows = []
                for ii in range(length):
                    key = (items[ii]['model'], items[ii]['iteration'], 
                           (items[ii]['top'], items[ii]['weighting_scheme']))
                    correct_predictions[key] += (
                        items[ii]['num_idx']==predicted_logits[ii,0].item())
                    total_predictions[key] += 1
                    rows.append((model_name, items[ii]['model'], items[ii]['imagenet_id'], 
                                 items[ii]['num_idx'], items[ii]['name'], 
                                 items[ii]['iteration'], items[ii]['top'], items[ii]['weighting_scheme'], 
                                 predicted_logits[ii, 0].item(), 
                                 *[predicted_logits[ii, iii].item() for iii in range(k)]))

                cursor.executemany(f"""
                        INSERT INTO predictions (pred_model, gen_model, imagenet_id, num_idx, name,
                            iteration, top, weighting, prediction
                            {"," if k > 0 else ""}
                            {", ".join(predictions)}) VALUES 
                        (?, ?, ?, ?, ?, ?, ?, ?, ?
                         {"," if k > 0 else ""}
                         {", ".join(["?" for _ in range(k)])})
                    """, rows
                )
            elif type(dataset) is ImagesByBlockDataset:
                rows = []
                for ii in range(length):
                    key = ((items[ii]['model'], items[ii]['block']), items[ii]['gen_type'],
                           items[ii]['iteration'])
                    correct_predictions[key] += (
                        items[ii]['num_idx']==predicted_logits[ii,0].item())
                    total_predictions[key] += 1
                    rows.append((model_name, items[ii]['model'], items[ii]['imagenet_id'], 
                                 items[ii]['num_idx'], items[ii]['name'], 
                                 items[ii]['iteration'], items[ii]['block'], items[ii]['gen_type'], 
                                 predicted_logits[ii, 0].item(), 
                                 *[predicted_logits[ii, iii].item() for iii in range(k)]))

                cursor.executemany(f"""
                        INSERT INTO predictions (pred_model, gen_model, imagenet_id, num_idx, name,
                            iteration, block, gen_type, prediction
                            {"," if k > 0 else ""}
                            {", ".join(predictions)}) VALUES 
                        (?, ?, ?, ?, ?, ?, ?, ?, ?
                         {"," if k > 0 else ""}
                         {", ".join(["?" for _ in range(k)])})
                    """, rows
                )

        connection.commit()
        for ii in range(length):
                del items[ii]['img_common']
                del items[ii]['img_xception']
                del items[ii]['img_vit_common']
                del items[ii]['img_vit_miil']
        del common_tensor_imgs, xception_tensor_imgs, vit_common_tensor_imgs, vit_miil_tensor_imgs,
        del predicted_logits, tensor_imgs
        torch.cuda.empty_cache()
    
    cursor.close()
    connection.close()
    dataset.transforms = transforms_cache

    if type(dataset) is MixedPredictiveImages.MixedPredictiveImagesDataset:
        return {
            k: (correct_predictions[k] / total_predictions[k]) if total_predictions[k] > 0 else 'No Predictions'
            for k in product(dataset.models, gen_types, dataset.iterations, dataset.weight_top_combs)
        }
    elif type(dataset) is MixedPredictiveImagesMacoDataset:
        return {
            k: (correct_predictions[k] / total_predictions[k]) if total_predictions[k] > 0 else 'No Predictions'
            for k in product(dataset.models, dataset.iterations, dataset.weight_top_combs)
        }


def plausibility_score(imagenet: ImageNetTrainDataset, result_paths: List[str],
                       datasets: List[Union[MixedPredictiveImages.MixedPredictiveImagesDataset,
                          MostPredictiveImages.MostPredictiveImagesDataset,
                          MostPredictiveImagesByBlock.MostPredictiveImagesByBlockDataset,
                          MixedPredictiveImagesMacoDataset,
                          ImagesByBlockDataset]],
                       batch_size: int=3, device: Optional[str]=None):
    """Prepare the calculation of plausibility score calculation by storing the feature embeddings
    of all elements in the dataset on disk

    Args:
        model (VisionTransformer): The vision transformer to get the feature embedding from 
        imagenet (ImageNetTrainDataset): The image net train dataset for the plausbility score computation.
        result_path (str): The path to store the result db under.
        dataset (Union[MixedPredictiveImages.MixedPredictiveImagesDataset,
                 MostPredictiveImages.MostPredictiveImagesDataset,
                 MostPredictiveImagesByBlock.MostPredictiveImagesByBlockDataset]): The dataset \
                 to compute the plausibility score of.
        batch_size (int, optional): The batch size to load data with. Defaults to 10.
        device (str, optional): The device to run the model on. Defaults to None.
    """

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models = {
        model_name: timm.create_model(model_name, pretrained=True).eval().to(device)
        for model_name in gen_models
    }

    transforms = {
        'common': get_transforms(models['vit_base_patch16_224']),
        'miil': get_transforms(models['vit_base_patch16_224_miil'])
    }

    # extractor = create_feature_extractor(model, ['fc_norm'])

    # imagenet_loader = DataLoader(imagenet, batch_size, shuffle=False, num_workers=10)
    # loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=10)

    imagenet_tf_cache = imagenet.transforms
    imagenet.transforms = transforms
    
    tf_caches = []
    for dataset in datasets:
        tf_caches.append(dataset.transforms)
        dataset.transforms = transforms

    for result_path in result_paths:
        if os.path.exists(result_path):
            raise NameError('The result ' + result_path + ' already exists')
    
    sqlite3.register_adapter(np.int64, int)
    sqlite3.register_adapter(np.int32, int)

    connections = [sqlite3.connect(result_path) for result_path in result_paths]
    cursors = [connections[i].cursor() for i in range(len(connections))]

    distance_sums = []
    distance_counts = []

    for i, dataset in enumerate(datasets):
        if type(dataset) is MixedPredictiveImages.MixedPredictiveImagesDataset:
            connections[i].execute(f"""
                CREATE TABLE IF NOT EXISTS distances (
                    gen_model TEXT,
                    imagenet_id TEXT,
                    num_idx INTEGER,
                    name TEXT,
                    iteration INTEGER,
                    top TEXT,
                    weighting TEXT,
                    gen_type TEXT,
                    distance INTEGER,
                    distance_path TEXT
                )
            """)

            gen_types = [dataset.gen_type] if dataset.gen_type != 'all' else ['div', 'clear']
            
            distance_sums.append({
                k: 0
                for k in product(dataset.models, gen_types, dataset.iterations, dataset.weight_top_combs)
            })
            distance_counts.append({
                k: 0
                for k in product(dataset.models, gen_types, dataset.iterations, dataset.weight_top_combs)
            })
        elif type(dataset) is MixedPredictiveImagesMacoDataset:
            connections[i].execute(f"""
                CREATE TABLE IF NOT EXISTS distances (
                    gen_model TEXT,
                    imagenet_id TEXT,
                    num_idx INTEGER,
                    name TEXT,
                    iteration INTEGER,
                    top TEXT,
                    weighting TEXT,
                    distance INTEGER,
                    distance_path TEXT
                )
            """)

            distance_sums.append({
                k: 0
                for k in product(dataset.models, dataset.iterations, dataset.weight_top_combs)
            })
            distance_counts.append({
                k: 0
                for k in product(dataset.models, dataset.iterations, dataset.weight_top_combs)
            })
        elif type(dataset) is ImagesByBlockDataset:
            connections[i].execute(f"""
                CREATE TABLE IF NOT EXISTS distances (
                    gen_model TEXT,
                    imagenet_id TEXT,
                    num_idx INTEGER,
                    name TEXT,
                    iteration INTEGER,
                    block INTEGER,
                    gen_type TEXT,
                    distance INTEGER,
                    distance_path TEXT
                )
            """)

            gen_types = dataset.gen_types
            
            distance_sums.append({
                k: 0
                for k in product(dataset.model_block_combs, gen_types, dataset.iterations)
            })
            distance_counts.append({
                k: 0
                for k in product(dataset.models, gen_types, dataset.iterations)
            })
        else:
            print('Not implemented')
            return 'Not implemented'

    # feature_mmap = np.memmap(mmap_path, dtype=float, mode='w+', shape=len(loader.dataset, model.embed_dim))

    for category in tqdm(imagenet.get_imagenet_classes()):

        imagenet.lock_category(category)
        [dataset.lock_category(category) for dataset in datasets]

        features = {
            model_name: torch.empty(
                len(imagenet), 
                models[model_name].patch_embed.num_patches*models[model_name].embed_dim).to(device)
            for model_name in gen_models
        }

        inds_to_path = {}

        for i in tqdm(range(0, len(imagenet), batch_size), desc=f'Train Images {category}', leave=False, position=-1):
            
            length = min(len(imagenet)-i, batch_size)
            items = [imagenet[ii+i] for ii in range(length)]

            tensor_images_common = torch.stack([items[ii]['img_common'] 
                                                for ii in range(length)], dim=0).to(device)
            tensor_images_miil = torch.stack([items[ii]['img_miil'] 
                                                for ii in range(length)], dim=0).to(device)
            
            for model_name in gen_models:

                result = models[model_name].forward_features(
                    tensor_images_miil if model_name.endswith('miil') else tensor_images_common)

                features[model_name][i:i+length, :] = (
                    result[:,1:,:].detach().flatten(start_dim=1))
                
                
            # feature_mmap
            for ii in range(length):
                inds_to_path[i + ii] = items[ii]['path']

            for ii in range(length):
                del items[ii]['img_common']
                del items[ii]['img_miil']
            del tensor_images_common, tensor_images_miil, result
            torch.cuda.empty_cache()
        # for i, items in enumerate(tqdm(imagenet_loader, desc=f'Train Images {category}')):
            
        #     tensor_images = items['img'].to(device)
        #     items_len = len(items['name'])
        #     start_ind = batch_size*i

        #     results = model.forward_features(tensor_images)
        #     print(results.shape)
        #     features[start_ind:start_ind+items_len, :, :] = results[:,1:,:].cpu()
        #     # feature_mmap
        #     for ii in range(items_len):
        #         inds_to_path[i + ii] = items['path'][ii]

        for ds_i, dataset in enumerate(datasets):
            for i in tqdm(range(0, len(dataset), batch_size), desc=f'Generated Images {category} Dataset {ds_i}', leave=False, position=-1):

                length = min(len(dataset)-i, batch_size)
                items = [dataset[ii+i] for ii in range(length)]

                tensor_images_common = [items[ii]['img_common'].to(device) for ii in range(length)]
                tensor_images_miil = [items[ii]['img_miil'].to(device) for ii in range(length)]

                mins_with_inds = {}
                for ii in range(length):

                    result = models[items[ii]['model']].forward_features((tensor_images_miil[ii] 
                        if model_name.endswith('miil') else tensor_images_common[ii]).unsqueeze(0))

                    mins_with_inds[ii] = torch.min(
                        torch.cdist(result[:,1:,:].detach().flatten(start_dim=1),
                                    features[items[ii]['model']], p=2), dim=-1)

                # mins, min_inds = torch.min(torch.cdist(results[:,1:,:].detach().cpu().flatten(start_dim=1),
                #                                        features, p=2), dim=-1)

                if type(dataset) is MixedPredictiveImages.MixedPredictiveImagesDataset:
                    rows = []
                    for ii, item in enumerate(items):
                        key = (item['model'], item['gen_type'], item['iteration'], 
                            (item['top'], item['weighting_scheme']))
                        distance_sums[ds_i][key] += mins_with_inds[ii][0][0].item()
                        distance_counts[ds_i][key] += 1
                        rows.append((item['model'], item['imagenet_id'], 
                                    item['num_idx'], item['name'], 
                                    item['iteration'], item['top'], item['weighting_scheme'], 
                                    item['gen_type'], 
                                    mins_with_inds[ii][0][0].item(), 
                                    inds_to_path[mins_with_inds[ii][1][0].item()]))

                    cursors[ds_i].executemany(f"""
                            INSERT INTO distances (gen_model, imagenet_id, num_idx, name,
                                iteration, top, weighting, gen_type, distance, distance_path) VALUES 
                            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, rows
                    )

                elif type(dataset) is MixedPredictiveImagesMacoDataset:
                    rows = []
                    for ii, item in enumerate(items):
                        key = (item['model'], item['iteration'], 
                            (item['top'], item['weighting_scheme']))
                        distance_sums[ds_i][key] += mins_with_inds[ii][0][0].item()
                        distance_counts[ds_i][key] += 1
                        rows.append((item['model'], item['imagenet_id'], 
                                    item['num_idx'], item['name'], 
                                    item['iteration'], item['top'], item['weighting_scheme'], 
                                    mins_with_inds[ii][0][0].item(), 
                                    inds_to_path[mins_with_inds[ii][1][0].item()]))

                    cursors[ds_i].executemany(f"""
                            INSERT INTO distances (gen_model, imagenet_id, num_idx, name,
                                iteration, top, weighting, distance, distance_path) VALUES 
                            (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, rows
                    )
                
                elif type(dataset) is ImagesByBlockDataset:
                    rows = []
                    for ii, item in enumerate(items):
                        key = ((item['model'], item['block']), item['gen_type'], item['iteration'])
                        distance_sums[ds_i][key] += mins_with_inds[ii][0][0].item()
                        distance_counts[ds_i][key] += 1
                        rows.append((item['model'], item['imagenet_id'], 
                                    item['num_idx'], item['name'], 
                                    item['iteration'], item['block'], item['gen_type'], 
                                    mins_with_inds[ii][0][0].item(), 
                                    inds_to_path[mins_with_inds[ii][1][0].item()]))

                    cursors[ds_i].executemany(f"""
                            INSERT INTO distances (gen_model, imagenet_id, num_idx, name,
                                iteration, block, gen_type, distance, distance_path) VALUES 
                            (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, rows
                    )
                
                for ii in range(length):
                    del items[ii]['img_common']
                    del items[ii]['img_miil']
                del tensor_images_miil, tensor_images_common, result, mins_with_inds
                torch.cuda.empty_cache()
            # for i, items in enumerate(tqdm(loader, desc=f'Generated Images {category}')):

            #     tensor_images = items['img'].to(device)
            #     items_len = len(items['name'])

            #     results = model.forward_features(tensor_images)

            #     mins, min_inds = torch.min(torch.cdist(results[:,1:,:].cpu(), features, p=2), dim=-1)

            #     for ii in range(items_len):
            #         if type(dataset) is MixedPredictiveImages.MixedPredictiveImagesDataset:
            #             rows = []
            #             for ii in range(items_len):
            #                 key = (items['model'][ii], items['gen_type'][ii], items['iteration'][ii].item(), 
            #                     (items['top'][ii], items['weighting_scheme'][ii]))
            #                 distance_sum[key] += mins[ii].item()
            #                 distance_count[key] += 1
            #                 rows.append((items['model'][ii], items['imagenet_id'][ii], 
            #                             items['num_idx'][ii].item(), items['name'][ii], 
            #                             items['iteration'][ii].item(), items['top'][ii], items['weighting_scheme'][ii], items['gen_type'][ii], 
            #                             mins[ii].item(), inds_to_path[min_inds[ii].item()]))

            #             cursor.executemany(f"""
            #                     INSERT INTO predictions (pred_model, gen_model, imagenet_id, num_idx, name,
            #                         iteration, top, weighting, gen_type, distance, distance_path) VALUES 
            #                     (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            #                 """, rows
            #             )

            connections[ds_i].commit()
        del features
        torch.cuda.empty_cache()
            
    [cursor.close() for cursor in cursors]
    [connection.close() for connection in connections]
    imagenet.transforms = imagenet_tf_cache
    
    for i, tf_cache in enumerate(tf_caches): 
        datasets[i].transforms = tf_cache

    imagenet.lock_category()
    [dataset.lock_category() for dataset in datasets]

    plausibilities = []

    for i, dataset in enumerate(datasets):
        if type(dataset) is MixedPredictiveImages.MixedPredictiveImagesDataset:
            plausibilities.append({
                k: (distance_sums[i][k] / distance_counts[i][k]) 
                if distance_counts[i][k] > 0 else 'No Predictions'
                for k in product(dataset.models, gen_types, dataset.iterations, dataset.weight_top_combs)
            })
        elif type(dataset) is MixedPredictiveImagesMacoDataset:
            plausibilities.append({
                k: (distance_sums[i][k] / distance_counts[i][k]) 
                if distance_counts[i][k] > 0 else 'No Predictions'
                for k in product(dataset.models, dataset.iterations, dataset.weight_top_combs)
            })
        elif type(dataset) is ImagesByBlock:
            plausibilities.append({
                k: (distance_sums[i][k] / distance_counts[i][k]) 
                if distance_counts[i][k] > 0 else 'No Predictions'
                for k in product(dataset.models, gen_types, dataset.iterations)
            })

    return plausibilities


def fid_precompute_class_distr(dataset: Union[ImageNetTrainDataset,ImageNetValDataset],
                               results_path: str, device: Optional[str]=None, batch_size: int=3,
                               show_progression: bool=True):
    
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model: timm.models.InceptionV3 = timm.create_model('inception_v3', pretrained=True).eval().to(device)
    transforms = get_transforms(model)
    # extractor = create_feature_extractor(model, ['fc_norm'])

    # imagenet_loader = DataLoader(imagenet, batch_size, shuffle=False, num_workers=10)
    # loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=10)

    tf_cache = dataset.transforms
    dataset.transforms = transforms

    if not os.path.isdir(results_path):
        print('Making output dir ' + results_path)
        os.makedirs(results_path)

    for category in tqdm(dataset.get_imagenet_classes(), desc='Categories', disable=not show_progression):
        
        os.makedirs(os.path.join(results_path, category))

        dataset.lock_category(category)
        embeddings = torch.empty(len(dataset), 2048).cpu()

        for i in tqdm(range(0, len(dataset), batch_size), desc=f'Dataset Images {category}', leave=False, 
                      disable=not show_progression, position=-1):
            
            length = min(len(dataset)-i, batch_size)
            items = [dataset[ii+i] for ii in range(length)]

            tensor_images = torch.stack([items[ii]['img'] for ii in range(length)], dim=0).to(device)

            result = model.global_pool(model.forward_features(tensor_images))
            embeddings[i:i+length] = result.detach().cpu()

            for item in items:
                del item['img']
            del tensor_images
            torch.cuda.empty_cache()

        cov = torch.cov(embeddings.T)
        mean = torch.mean(embeddings, dim=0)

        torch.save(cov, os.path.join(results_path, category, 'cov.pt'))
        torch.save(mean, os.path.join(results_path, category, 'mean.pt'))


    dataset.transforms = tf_cache
    dataset.lock_category()

# partially adapted from https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
def fid_score(dataset: Union[MixedPredictiveImages.MixedPredictiveImagesDataset,
                MostPredictiveImages.MostPredictiveImagesDataset,
                MostPredictiveImagesByBlock.MostPredictiveImagesByBlockDataset,
                MixedPredictiveImagesMacoDataset,
                ImagesByBlockDataset], results_path: str, device: Optional[str]=None,
                batch_size: int=3, show_progression: bool=True,
                store_path: Optional[str]=None):
    """Compute the fid scores after having precomputed the imagenet component of the score.

    Args:
        dataset (Union[MixedPredictiveImages.MixedPredictiveImagesDataset, MostPredictiveImages.MostPredictiveImagesDataset, MostPredictiveImagesByBlock.MostPredictiveImagesByBlockDataset, MixedPredictiveImagesMacoDataset]): The dataset to compute the FID score on for every imagenet category
        results_path (str): The directory the precomputed imagenet means and covariances are placed.
        device (Optional[str], optional): The device to run inference on. Defaults to None.
        batch_size (int, optional): The batch size to run inference with. Defaults to 10.
        show_progression (bool, optional): True if tqdm progression should be shown. Defaults to True.
        store_path (Optional[str], optional): The path to store the result under in json format. Defaults to None.

    Raises:
        FileNotFoundError: If the directory of the precomputations or the precomputations do not exist.
        ValueError: If the multiplied covariance matrix root contains large imaginary components.
    """

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model: timm.models.InceptionV3 = timm.create_model('inception_v3', pretrained=True).eval().to(device)
    transforms = get_transforms(model)

    tf_cache = dataset.transforms
    dataset.transforms = transforms

    scores = {}

    loader = DataLoader(dataset, batch_size, False, num_workers=7)

    if not os.path.isdir(results_path):
        raise FileNotFoundError(f'The folder {results_path} does not exist')

    for category in tqdm(dataset.get_imagenet_classes(), desc='Categories', disable=not show_progression):
        
        dir = os.path.join(results_path, category)

        cov_imgnet = torch.load(os.path.join(dir, 'cov.pt'))
        mean_imgnet = torch.load(os.path.join(dir, 'mean.pt'))

        dataset.lock_category(category)
        embeddings = torch.empty(len(dataset), 2048).cpu()

        # for i, items in tqdm(enumerate(loader), desc=f'Dataset Images {category}', leave=False, 
        #               disable=True, position=-1):
            
        #     tensor_images = items['img'].to(device)

        #     result = model.global_pool(model.forward_features(tensor_images))
        #     embeddings[i:i+length] = result.detach().cpu()

        #     for item in items:
        #         del item['img']
        #     del tensor_images
        #     torch.cuda.empty_cache()
        for i in tqdm(range(0, len(dataset), batch_size), desc=f'Dataset Images {category}', leave=False, disable=True, position=-1):
            
            length = min(len(dataset)-i, batch_size)
            items = [dataset[ii+i] for ii in range(length)]

            tensor_images = torch.stack([items[ii]['img'] for ii in range(length)], dim=0).to(device)

            result = model.global_pool(model.forward_features(tensor_images))
            embeddings[i:i+length] = result.detach().cpu()

            for item in items:
                del item['img']
            del tensor_images
            torch.cuda.empty_cache()

        cov = torch.cov(embeddings.T)
        mean = torch.mean(embeddings, dim=0)

        epsilon = 1e-6

        mean_diff = mean - mean_imgnet

        cov_np = cov.numpy()
        cov_imgnet_np = cov_imgnet.numpy()

        variances, _ = linalg.sqrtm(cov_np @ cov_imgnet_np, disp=False)
        if not np.isfinite(variances).all():
            offset = np.eye(variances.shape[0]) * epsilon
            variances = linalg.sqrtm((cov_np + offset) @ (cov_imgnet_np + offset))

        if np.iscomplexobj(variances):
            # if not np.allclose(np.diagonal(variances).imag, 0, atol=1e-3):
            #     raise ValueError(f'Imaginary component too large {np.max(np.abs(variances.imag))}')
            variances = variances.real

        scores[category] = mean_diff.dot(mean_diff).item() + torch.trace(cov).item() + \
            torch.trace(cov_imgnet).item() - 2 * np.trace(variances)
            
    dataset.transforms = tf_cache
    dataset.lock_category()

    if store_path is not None:
        with open(store_path, 'x') as f:
            json.dump(scores, f)

    return scores

# from https://github.com/pytorch/pytorch/issues/25481
def torch_sqrtm(matrix):
    """Compute the square root of a positive definite matrix."""
    _, s, v = matrix.svd()
    good = s > s.max(-1, True).values * s.size(-1) * torch.finfo(s.dtype).eps
    components = good.sum(-1)
    common = components.max()
    unbalanced = common != components.min()
    if common < s.size(-1):
        s = s[..., :common]
        v = v[..., :common]
        if unbalanced:
            good = good[..., :common]
    if unbalanced:
        s = s.where(good, torch.zeros((), device=s.device, dtype=s.dtype))
    return (v * s.sqrt().unsqueeze(-2)) @ v.transpose(-2, -1)