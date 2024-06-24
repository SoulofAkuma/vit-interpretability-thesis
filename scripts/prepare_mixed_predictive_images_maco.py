import os
import shutil
import argparse
from typing import List
from itertools import product

DEFAULT_ITERATIONS = ['1000']
DEFAULT_MODELS = [
    'vit_base_patch16_224',
    'vit_base_patch32_224',
    'vit_large_patch16_224',
    'vit_base_patch16_224_miil'
]
DEFAULT_TOPS = [
    '1',
    '1,2,3,4'
]
WEIGHTING_SCHEMES = {
    'weighted': 'softmax'
}

def prepare_data(imgs_path: str, iterations: List[str],
                 models: List[str], tops: List[str]):
    
    subfolders = product(models, iterations, map(lambda x: f'top_{x}', tops), 
                         [*list(WEIGHTING_SCHEMES.values()), 'unweighted'])
    for model, iteration, top, weighting_scheme in subfolders:
        os.makedirs(os.path.join(imgs_path, model, iteration, top, weighting_scheme), exist_ok=True)

    for model in models:
        _, _, files = next(os.walk(os.path.join(imgs_path, model)))

        for file in files:
            
            file_name, ext = os.path.splitext(file)
            file_path = os.path.join(imgs_path, model, file)

            fn_parts = file_name.split('_')
            batch_ind = None
            weighted = ''
            if len(fn_parts) == 5:
                cls, _, _, top, iteration = fn_parts
            elif len(fn_parts) == 6:
                cls, _, _, top, weighted, iteration = fn_parts
            else:
                raise NameError('Invalid file name ' + file_name)

            top = 'top_' + top

            new_img_name = f'{cls}{("_" + batch_ind) if batch_ind is not None else ""}{ext}'
            new_img_path = os.path.join(imgs_path, model, iteration, top,
                                        WEIGHTING_SCHEMES.get(weighted, 'unweighted'), 
                                        new_img_name)
            
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
    parser.add_argument('--tops', default=DEFAULT_TOPS, nargs='+',
                        required=False, type=str)
    args = parser.parse_args()
    prepare_data(args.images_path, args.iterations, args.models, args.tops)