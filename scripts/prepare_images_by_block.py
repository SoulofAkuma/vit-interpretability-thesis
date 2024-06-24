import os
import shutil
import argparse
from typing import List
from itertools import product

DEFAULT_ITERATIONS = ['500']
BLOCKS_BY_MODEL = {
    'vit_base_patch16_224': 12,
    'vit_base_patch32_224': 12,
    'vit_large_patch16_224': 24,
    'vit_base_patch16_224_miil': 12
}

DEFAULT_MODELS = [
    'vit_base_patch16_224',
    'vit_base_patch32_224',
    'vit_large_patch16_224',
    'vit_base_patch16_224_miil'
]

DEFAULT_CATEGORIES = [
    'cls_token', 'clear'
]

def prepare_data(imgs_path: str, iterations: List[str], categories: List[str],
                 models: List[str]):
    
    block_dirs_by_model = {
        model_name: [f'block_{i}' for i in range(BLOCKS_BY_MODEL[model_name])]
        for model_name in models
    }

    subfolders = product(models, categories, iterations)
    for model, cat, iteration in subfolders:
        for block_dir in block_dirs_by_model[model]:
            os.makedirs(os.path.join(imgs_path, model, cat, iteration, block_dir), 
                        exist_ok=True)

    for model in models:
        _, _, files = next(os.walk(os.path.join(imgs_path, model)))

        for file in files:
            
            file_name, ext = os.path.splitext(file)
            file_path = os.path.join(imgs_path, model, file)

            fn_parts = file_name.split('_')
            
            if len(fn_parts) == 5:
                cls, cat, _, block_ind, iteration = fn_parts
            elif len(fn_parts) == 6:
                cls, cat, _, _, block_ind, iteration = fn_parts
            elif len(fn_parts) == 7:
                cls, cat, _, block_ind, _, _, iteration = fn_parts
            elif len(fn_parts) == 8:
                cls, cat, _, _, block_ind, _, _, iteration = fn_parts
            else:
                raise NameError('Invalid file name ' + file_name)

            new_img_name = f'{cls}{ext}'
            new_img_path = os.path.join(imgs_path, model, 
                                        cat if cat == 'clear' else 'cls_token', 
                                        iteration, f'block_{block_ind}', new_img_name)
            
            # print(file_path, new_img_path)
            shutil.move(file_path, new_img_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-path', 
                        default=os.path.join(os.path.dirname(__file__), 'images'),
                        type=str, required=True)
    parser.add_argument('--iterations', default=DEFAULT_ITERATIONS,
                        nargs='+', required=False, type=str)
    parser.add_argument('--models', default=DEFAULT_MODELS, nargs='+', 
                        required=False, type=str)
    parser.add_argument('--categories', default=DEFAULT_CATEGORIES,
                        nargs='+', required=False, type=str)
    args = parser.parse_args()
    prepare_data(args.images_path, args.iterations, args.categories, args.models)