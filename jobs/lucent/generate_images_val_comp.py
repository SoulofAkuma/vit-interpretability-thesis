from timm import create_model
import src.visualizers.lucent as vl
import src.visualizers.maco_lucent as vm
from lucent.optvis import render
import lucent.optvis.param as param
from lucent.optvis import transform
import numpy as np
import torch

def generate_images(model_name, image_size_lucid, image_size_maco, model_img_size, 
                    thresholds_lucid, thresholds_maco, classes, device):

    model = create_model(model_name, pretrained=True).to(device).eval()

    results_lucid = {}
    results_maco = {}

    for cls in classes.keys():
        block = classes[cls]['block']
        index = classes[cls]['index']

        for img_ind in range(10):
            objective = vl.key_neuron_objective(block, index)

            param_f_clear = lambda: param.image(image_size_lucid, device=device)

            transforms = transform.standard_transforms_for_device(device).copy()
            transforms.append(torch.nn.Upsample(size=model_img_size, mode='bilinear', align_corners=True))

            clear_result = render.render_vis(model, objective, param_f_clear, transforms=transforms,
                                            thresholds=thresholds_lucid, show_image=False, show_inline=False, 
                                            device=device)
            
            for i, img in enumerate(clear_result):
                results_lucid[f'{cls}_clear_{thresholds_lucid[i]}_{img_ind}_lucid'] = (img * 255).astype(np.uint8)

            objective = vm.key_neuron_objective(block, index)

            param_f = vm.maco_lucent_param_f(image_size_maco, (-2.5, 2.5), device, 1.0)
            optimizer = vm.maco_optimizer_generator(1.0)
            alpha_retriever, post_grad_f = vm.maco_post_grad_f(image_size_maco, device)
            transforms = vm.maco_transforms(6, (0.20, 0.25), 0.05, model_img_size, image_size_maco, device)

            result_imgs = render.render_vis(model, objective, param_f, optimizer, transforms, thresholds_maco,
                            preprocess=False, show_image=False, show_inline=False,
                            fixed_image_size=model_img_size, device=device, 
                            post_grad_f=post_grad_f, skip_size_transform=True,
                            images_as_tensor=True, progress=True)
            alpha = alpha_retriever()

            for i, img in enumerate(result_imgs):
                img = vm.image_posttasks(img, alpha, 0.01, 0.8)
                results_maco[f'{cls}_clear_{thresholds_maco[i]}_{img_ind}_maco'] = (img.numpy() * 255).astype(np.uint8)

        
    return results_lucid, results_maco