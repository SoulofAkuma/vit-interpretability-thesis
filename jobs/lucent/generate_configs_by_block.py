from argparse import ArgumentParser
import os
import json
from timm import create_model
import torch
from src.utils.extraction import extract_value_vectors
from src.utils.model import embedding_projection
from src.analyzers.vector_analyzer import most_predictive_ind_for_classes_by_block
from typing import Dict, List, Tuple
import torch.nn.functional as F

MODELS = [
    'vit_base_patch16_224',
    'vit_base_patch32_224',
    'vit_base_patch16_224_miil',
    'vit_large_patch16_224',
]
MODEL_IMG_SIZE = 224
IMAGE_SIZE = 224
THRESHOLDS = [500,]

def create_configs(dir: str):

    os.makedirs(dir, exist_ok=True)

    with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/data/imagenet_class_index.json'))
              , 'r') as file:
        mapping = json.load(file)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    configs = [[] for i in range(16)]

    for MODEL in MODELS:

        print('Generating config for model ' + MODEL)
        model = create_model(MODEL, pretrained=True).to(device).eval()

        values = extract_value_vectors(model, device=device)
        embedded_values = embedding_projection(model, values, device=device).to(device)
        most_pred_inds = most_predictive_ind_for_classes_by_block(embedded_values, device=device)

        # 1000 / 63 = 15.625 => 16 configs for 16 gpus
        if MODEL == 'vit_large_patch16_224':
            for i in range(0, 1000, 63):
                # name of the generation, list of block index, row index, weight
                imagenet_classes: Dict[str, List[Tuple[int, int, float]]] = {}

                # 64 classes per gpu
                for ii in range(45, 63):
                    if i + ii >= 1000:
                        break
                    inds_for_cls = most_pred_inds[:,i+ii].tolist()
                    
                    imagenet_classes[mapping[str(i + ii)][0]] = {
                        f'block_{iii}': [iii, inds_for_cls[iii]] 
                        for iii in range(len(model.blocks))
                    }

                configs[i//63].append({
                        'model': MODEL,
                        'model_img_size': MODEL_IMG_SIZE,
                        'classes': imagenet_classes,
                        'image_size': IMAGE_SIZE,
                        'thresholds': THRESHOLDS,
                    })
                print(f'Config for {MODEL} generated' )
        else:
            for i in range(0, 1000, 63):
                # name of the generation, list of block index, row index, weight
                imagenet_classes: Dict[str, List[Tuple[int, int, float]]] = {}

                # 64 classes per gpu
                for ii in range(0, 63):
                    if i + ii >= 1000:
                        break
                    inds_for_cls = most_pred_inds[:,i+ii].tolist()
                    
                    imagenet_classes[mapping[str(i + ii)][0]] = {
                        f'block_{i}': [i, inds_for_cls[i]] for i in range(len(model.blocks))
                    }

                configs[i//63].append({
                        'model': MODEL,
                        'model_img_size': MODEL_IMG_SIZE,
                        'classes': imagenet_classes,
                        'image_size': IMAGE_SIZE,
                        'thresholds': THRESHOLDS,
                    })
                print(f'Config for {MODEL} generated' )

    for i, config in enumerate(configs):
        filepath = os.path.join(dir, f'config_{i}.json')
        with open(filepath, 'w+') as file:
            print(f'Writing config {i} to {filepath}')
            json.dump(config, file)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output-dir', type=str)

    args = parser.parse_args()
    create_configs(os.path.abspath(args.output_dir))