from timm import create_model
from src.visualizers.lucent import key_neuron_objective
from lucent.optvis import render
import lucent.optvis as optvis
import lucent.optvis.param as param
from lucent.optvis import transform
import numpy as np
import torch
from PIL import Image
import os



def generate_images(model_name, image_size, model_img_size, thresholds, classes, 
                    device, results_path):
    
    os.makedirs(os.path.join(results_path, model_name), exist_ok=True)
    
    def save_image(img_name, img):
        img = (img * 255).astype(np.uint8)

        if len(img.shape) > 3:
            Image.fromarray(img[0]).save(os.path.join(results_path, model_name, f'{img_name}.png'))
        else:
            Image.fromarray(img).save(os.path.join(results_path, model_name, f'{img_name}.png'))

    model = create_model(model_name, pretrained=True).to(device).eval()

    results = {}

    for cls in classes.keys():

        for img_name in classes[cls].keys():


            neuron = classes[cls][img_name]

            objective = key_neuron_objective(neuron[0], neuron[1])
            cls_token_objective = key_neuron_objective(neuron[0], neuron[1], token_boundaries=(0,1))

            param_f_clear = lambda: param.image(image_size, device=device)
            param_f_cls_token = lambda: param.image(image_size, device=device)

            transforms = transform.standard_transforms_for_device(device).copy()
            transforms.append(torch.nn.Upsample(size=model_img_size, mode='bilinear', align_corners=True))

            clear_result = render.render_vis(model, objective, param_f_clear, transforms=transforms,
                                             thresholds=thresholds, show_image=False, show_inline=False, 
                                             device=device)
            
            transforms = transform.standard_transforms_for_device(device).copy()
            transforms.append(torch.nn.Upsample(size=model_img_size, mode='bilinear', align_corners=True))
            
            cls_token_result = render.render_vis(model, cls_token_objective, param_f_cls_token, 
                                                 transforms=transforms, thresholds=thresholds, 
                                                 show_image=False, show_inline=False, device=device)
            
            print(f'Generated block {neuron[0]} for {cls}')

            for i, img in enumerate(clear_result):
                save_image(f'{cls}_clear_{img_name}_block_{neuron[0]}_{thresholds[i]}', img)

            for i, img in enumerate(cls_token_result):
                save_image(f'{cls}_cls_token_{img_name}_block_{neuron[0]}_{thresholds[i]}', img)

    return results