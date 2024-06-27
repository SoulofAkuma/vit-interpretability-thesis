import os
import json
import torch
from generate_images_val_comp import generate_images
from PIL import Image
import time

job_index = os.environ['SLURM_PROCID']
node_name = os.environ['SLURMD_NODENAME']
job_id = os.environ['SLURM_JOB_ID']
# job_index = 0
# node_name = ''

CONFIG_PATH = '/scratch/vihps/vihps01/vit-interpretability-thesis/configs-val-comp'
RESULTS_PATH = '/scratch/vihps/vihps01/vit-interpretability-thesis/images-val-comp-' + str(job_id)
RESULT_STATS_PATH = '/scratch/vihps/vihps01/vit-interpretability-thesis/job-reports'
# CONFIG_PATH = 'A:\\My Drive\\Unviersity\\Thesis\\repo\\configs-val-comp'
# RESULTS_PATH = 'A:\\My Drive\\Unviersity\\Thesis\\repo\\images-val-comp'
# RESULT_STATS_PATH = 'A:\\My Drive\\Unviersity\\Thesis\\repo\\job-reports'

os.makedirs(CONFIG_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(RESULT_STATS_PATH, exist_ok=True)


os.environ['MIOPEN_USER_DB_PATH'] = f'/scratch/vihps/vihps01/.config/miopen_{job_index}/'

config = None
with open(os.path.join(CONFIG_PATH, f'config_{job_index}.json'), 'r') as file:
    config = json.load(file)

device_count = torch.cuda.device_count()
device = torch.device(f'cuda:{int(job_index) % device_count}'
                      if torch.cuda.is_available() else 'cpu')


start_time = time.time()

images_generated = {}

for model_config in config:

    images_lucid, images_maco = generate_images(model_config['model'], 
                                                model_config['image_size_lucid'], 
                                                model_config['image_size_maco'], 
                                                model_config['model_img_size'], 
                                                [model_config['thresholds_lucid']], 
                                                [model_config['thresholds_maco']], 
                                                model_config['classes'], device)

    images_generated[model_config['model']] = [*list(images_lucid.keys()), *list(images_maco.keys())]

    os.makedirs(os.path.join(RESULTS_PATH, model_config['model'], 'maco'), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_PATH, model_config['model'], 'lucid'), exist_ok=True)

    for img_name in images_lucid.keys():
        if len(images_lucid[img_name].shape) > 3:
            Image.fromarray(images_lucid[img_name][0]).save(os.path.join(RESULTS_PATH, model_config['model'], 'lucid', f'{img_name}.png'))
        else:
            Image.fromarray(images_lucid[img_name]).save(os.path.join(RESULTS_PATH, model_config['model'], 'lucid', f'{img_name}.png'))

    for img_name in images_maco.keys():
        if len(images_maco[img_name].shape) > 4:
            Image.fromarray(images_maco[img_name][0], 'RGBA').save(os.path.join(RESULTS_PATH, model_config['model'], 'maco', f'{img_name}.png'))
        else:
            Image.fromarray(images_maco[img_name], 'RGBA').save(os.path.join(RESULTS_PATH, model_config['model'], 'maco', f'{img_name}.png'))


with open(os.path.join(RESULT_STATS_PATH, f'results_{job_index}.json'), 'w+') as file:
    json.dump({
        'job_index': job_index,
        'node': node_name,
        'execution_time': time.time() - start_time,
        'images_generated': images_generated
    }, file)