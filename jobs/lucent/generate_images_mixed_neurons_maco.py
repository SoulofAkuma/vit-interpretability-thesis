from timm import create_model
from src.visualizers.maco_lucent import key_neuron_objective, maco_lucent_param_f, maco_optimizer_generator, maco_post_grad_f, maco_transforms, image_posttasks
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

            neurons = classes[cls][img_name]

            objective = neurons[0][2] * key_neuron_objective(neurons[0][0], neurons[0][1])

            for i in range(1, len(neurons)):
                objective += neurons[i][2] * key_neuron_objective(neurons[i][0], neurons[i][1])

            param_f = maco_lucent_param_f(image_size, (-2.5, 2.5), device, 1.0)
            optimizer = maco_optimizer_generator(1.0)
            alpha_retriever, post_grad_f = maco_post_grad_f(image_size, device)
            transforms = maco_transforms(6, (0.20, 0.25), 0.05, model_img_size, image_size, device)

            result_imgs = render.render_vis(model, objective, param_f, optimizer, transforms, thresholds,
                            preprocess=False, show_image=False, show_inline=False,
                            fixed_image_size=model_img_size, device=device, 
                            post_grad_f=post_grad_f, skip_size_transform=True,
                            images_as_tensor=True, progress=False)
            alpha = alpha_retriever()

            print(f'Generated {img_name} for {cls}')
            
            for i, img in enumerate(result_imgs):
                img = image_posttasks(img, alpha, 0.01, 0.8)
                results[f'{cls}_maco_{img_name}_{thresholds[i]}'] = (img.numpy() * 255).astype(np.uint8)

    return results