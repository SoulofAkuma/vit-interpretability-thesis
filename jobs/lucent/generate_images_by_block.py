from timm import create_model
from src.visualizers.lucent import key_neuron_objective
from lucent.optvis import render
import lucent.optvis as optvis
import lucent.optvis.param as param
from lucent.optvis import transform
import numpy as np
import torch

def generate_images(model_name, image_size, model_img_size, thresholds, classes, device):

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
            
            cls_token_result = render.render_vis(model, cls_token_objective, param_f_clear, 
                                                 transforms=transforms, thresholds=thresholds, 
                                                 show_image=False, show_inline=False, device=device)
            
            for i, img in enumerate(clear_result):
                results[f'{cls}_clear_{img_name}_{thresholds[i]}'] = (img * 255).astype(np.uint8)

            for i, img in enumerate(cls_token_result):
                results[f'{cls}_cls_token_{img_name}_{thresholds[i]}'] = (img * 255).astype(np.uint8)

    return results