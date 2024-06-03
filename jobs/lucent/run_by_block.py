import os
import json
import torch
from generate_images_by_block import generate_images
from PIL import Image
import time

CONFIG_PATH = '/scratch/vihps/vihps01/vit-interpretability-thesis/configs-by-block'
RESULTS_PATH = '/scratch/vihps/vihps01/vit-interpretability-thesis/images-by-block'
RESULT_STATS_PATH = '/scratch/vihps/vihps01/vit-interpretability-thesis/job-reports'

os.makedirs(CONFIG_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(RESULT_STATS_PATH, exist_ok=True)

job_index = os.environ['SLURM_PROCID']
node_name = os.environ['SLURMD_NODENAME']

os.environ['MIOPEN_USER_DB_PATH'] = f'/scratch/vihps/vihps01/vit-interpretability-thesis/.config/miopen_{job_index}/'
os.environ['MIOPEN_DEBUG_DISABLE_SQL_WAL'] = '1'

config = None
with open(os.path.join(CONFIG_PATH, f'config_{job_index}.json'), 'r') as file:
    config = json.load(file)

device_count = torch.cuda.device_count()
device = torch.device(f'cuda:{int(job_index) % device_count}'
                      if torch.cuda.is_available() else 'cpu')


start_time = time.time()

images_generated = {}

for model_config in config:

    images = generate_images(model_config['model'], model_config['image_size'], 
                             model_config['model_img_size'], model_config['thresholds'], 
                             model_config['classes'], device)

    images_generated[model_config['model']] = list(images.keys())

    os.makedirs(os.path.join(RESULTS_PATH, model_config['model']), exist_ok=True)

    for img_name in images.keys():
        if len(images[img_name].shape) > 3:
            Image.fromarray(images[img_name][0]).save(os.path.join(RESULTS_PATH, model_config['model'], f'{img_name}.png'))
        else:
            Image.fromarray(images[img_name]).save(os.path.join(RESULTS_PATH, model_config['model'], f'{img_name}.png'))

with open(os.path.join(RESULT_STATS_PATH, f'results_{job_index}.json'), 'w+') as file:
    json.dump({
        'job_index': job_index,
        'node': node_name,
        'execution_time': time.time() - start_time,
        'images_generated': images_generated
    }, file)