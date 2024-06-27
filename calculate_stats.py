import timm
import timm.models.vision_transformer
import torch
from src.datasets.ImagesByBlock import ImagesByBlockDataset
from src.utils.transformation import get_transforms
from genericpath import isfile
from src.datasets.ImageNet import ImageNetValDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import sqlite3
import os
from torchvision.models.feature_extraction import create_feature_extractor
from src.utils.extraction import extract_value_vectors
from src.utils.model import embedding_projection
from src.analyzers.vector_analyzer import k_most_predictive_ind_for_classes
import torch.nn.functional as F
from pathlib import Path
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from src.datasets.MixedPredictiveImages import MixedPredictiveImagesDataset
from src.datasets.MixedPredictiveImagesMaco import MixedPredictiveImagesMacoDataset
from src.datasets.ImageNet import ImageNetTrainDataset
from src.analyzers.visualization_analyzer import transferability_score
from src.analyzers.visualization_analyzer import plausibility_score
from src.analyzers.visualization_analyzer import fid_precompute_class_distr
from src.analyzers.visualization_analyzer import fid_score

imagenet = ImageNetTrainDataset('A:\\CVData\\ImageNet')
dataset_maco = MixedPredictiveImagesMacoDataset(
    'A:\\CVData\\project-images\\images-mixed-1280-fc1-multimodel-maco')
dataset_mixed = MixedPredictiveImagesDataset(
    'A:\\CVData\\project-images\\images-mixed-224-fc1-multimodel')
dataset_by_block = ImagesByBlockDataset(
    'A:\\CVData\\project-images\\images-by-block-224-fc1')
    
# print('Transferability maco')
# transferability = transferability_score(
#     dataset_maco,
#     'A:\\CVData\\project-images\\images-mixed-1280-fc1-multimodel-maco\\transferability_with_gen.db', 
#     k=10, device=device, include_gen_models=True, batch_size=3)

# print('Plausibility maco and mixed')
# plausibility = plausibility_score(
#     imagenet, 
#     ['A:\\CVData\\project-images\\images-mixed-1280-fc1-multimodel-maco\\plausibility_with_gen.db',
#      'A:\\CVData\\project-images\\images-mixed-224-fc1-multimodel\\plausibility_with_gen.db'], [dataset_maco, dataset_mixed], device=device)


# print('Transferability mixed')
# transferability = transferability_score(
#     dataset_mixed,
#     'A:\\CVData\\project-images\\images-mixed-224-fc1-multimodel\\transferability_with_gen.db', 
#     k=10, device=device, include_gen_models=True, batch_size=5)

# TODO print('Plausibility mixed')
# plausibility = plausibility_score(imagenet, 'A:\\CVData\\project-images\\images-mixed-224-fc1-multimodel\\plausibility_with_gen.db', dataset_mixed, device=device)

# print('Transferability by block')
# transferability = transferability_score(
#     dataset_by_block,
#     'A:\\CVData\\project-images\\images-by-block-224-fc1\\transferability_with_gen.db', 
#     k=10, device=device, include_gen_models=True, batch_size=3)

imgnet_val = ImageNetValDataset('A:\\CVData\\ImageNet')
# fid_precompute_class_distr(imgnet_val, 'A:\\CVData\\ImageNet\\val_fid_distr', device)


model_names = [
    'vit_base_patch16_224_miil', 
    'vit_large_patch16_224',
    'vit_base_patch32_224', 
    'vit_base_patch16_224', 
]
# models = {
#     model_name: timm.create_model(model_name, pretrained=True).eval().to(device)
#     for model_name in model_names
# }
# model_transforms = {
#     'miil': get_transforms(models['vit_base_patch16_224_miil']),
#     'vanilla': get_transforms(models['vit_base_patch16_224'])
# }
# transform_keys = {
#     model_name: 'miil' if model_name.find('miil') != -1 else 'vanilla'
#     for model_name in model_names 
# }

random_images = np.tile(np.arange(0, 50, 1, dtype=np.uint8), (1000, 1))
random_images = np.random.default_rng().permuted(random_images, axis=1)
random_images = random_images[:,:5]

# k_most_pred_by_model = {
#     model_name: k_most_predictive_ind_for_classes(
#         embedding_projection(models[model_name], extract_value_vectors(models[model_name], device), device),
#         10, device)
#     for model_name in model_names
# }

# mlp_fc2_biases = {
#     model_name: [models[model_name].blocks[i].mlp.fc2.bias.detach() 
#                  for i in range(len(models[model_name].blocks))]
#     for model_name in model_names
# }
# mlp_fc2_biases_l2 = {
#     model_name: [(mlp_fc2_biases[model_name][i] ** 2).sum().sqrt().item() 
#                  for i in range(len(mlp_fc2_biases[model_name]))]
#     for model_name in model_names
# }
# mlp_fc2_weights = {
#     model_name: [models[model_name].blocks[i].mlp.fc2.weight.T.detach()
#                  for i in range(len(models[model_name].blocks))]
#     for model_name in model_names
# }
# model_heads = {
#     model_name: models[model_name].head.eval()
#     for model_name in model_names
# }
# model_norms = {
#     model_name: models[model_name].norm.eval()
#     for model_name in model_names
# }
# mlp_fc2_biases_pred = {
#     model_name: [model_heads[model_name](model_norms[model_name](bias)) 
#                  for bias in mlp_fc2_biases[model_name]]
#     for model_name in model_names
# }

sqlite3.register_adapter(np.int64, int)
sqlite3.register_adapter(np.int32, int)

def create_pred_db(path, results_path, bias_pl_resid_path, bias_pl_vec_noise_path, bias_pl_all_path,
                   cls_token_before_most_predictive_pred_path, 
                   cls_token_after_most_predictive_pred_path):

    dataset = ImageNetValDataset('A:\\CVData\\ImageNet', transforms=model_transforms)
    batch_size = 3
    k = 10

    connection = sqlite3.connect(path)
    cursor = connection.cursor()
    connection.execute(f"""
        CREATE TABLE IF NOT EXISTS predictions (
            path TEXT,
            pred_model TEXT,
            imagenet_id TEXT,
            num_idx INTEGER,
            name TEXT,
            {','.join([f'top_{i}_score REAL' for i in range(k)])},
            {','.join([f'top_{i}_ind INTEGER' for i in range(k)])},
            sum_row REAL,
            l1_row REAL,
            l2_row REAL,
            exp_reciprocal REAL,
            min_row REAL,
            pred_path TEXT
        )
    """)
    connection.execute(f"""
        CREATE TABLE IF NOT EXISTS vec_activations (
            path TEXT,
            pred_model TEXT,
            imagenet_id TEXT,
            num_idx INTEGER,
            name TEXT,
            {','.join([f'top_{i}_cls_token_act REAL' for i in range(k)])},
            {','.join([f'top_{i}_img_token_avg_act REAL' for i in range(k)])},
            {','.join([f'top_{i}_block_ind INTEGER' for i in range(k)])},
            {','.join([f'top_{i}_vec_ind INTEGER' for i in range(k)])},
            {','.join([f'top_{i}_max_cls_token_act REAL' for i in range(k)])},
            {','.join([f'top_{i}_max_cls_act_block_ind INTEGER' for i in range(k)])},
            {','.join([f'top_{i}_max_cls_act_vec_ind INTEGER' for i in range(k)])},
            {','.join([f'top_{i}_max_img_token_avg_act REAL' for i in range(k)])},
            {','.join([f'top_{i}_max_img_avg_act_block_ind INTEGER' for i in range(k)])},
            {','.join([f'top_{i}_max_img_avg_act_vec_ind INTEGER' for i in range(k)])},
            top_bias_l2 REAL,
            top_bias_pl_res_l2 REAL,
            resid_l2 REAL,
            top_bias_pl_vec_noise_l2 REAL,
            vec_noise_l2 REAL,
            top_bias_pl_all_l2 REAL,
            all_l2 REAL,
            path_bias_pl_res_pred TEXT,
            path_bias_pl_vec_noise_pred TEXT,
            path_bias_pl_all_pred TEXT,
            mean_bias_pl_res_pred REAL,
            mean_bias_pl_vec_noise_pred REAL,
            mean_bias_pl_all_pred REAL,
            std_bias_pl_res_pred REAL,
            std_bias_pl_vec_noise_pred REAL,
            std_bias_pl_all_pred REAL,
            max_bias_pl_res_pred REAL,
            max_bias_pl_vec_noise_pred REAL,
            max_bias_pl_all_pred REAL,
            mean_top_bias_pred REAL,
            std_top_bias_pred REAL,
            max_top_bias_pred REAL,
            mean_resid_pred REAL,
            std_resid_pred REAL,
            max_resid_pred REAL,
            mean_vec_noise_pred REAL,
            std_vec_noise_pred REAL,
            max_vec_noise_pred REAL,
            mean_all_pred REAL,
            std_all_pred REAL,
            max_all_pred REAL,
            cls_score_bias_pl_res_pred REAL,
            cls_score_bias_pl_vec_noise_pred REAL,
            cls_score_bias_pl_all_pred REAL,
            cls_score_top_bias_pred REAL,
            cls_score_resid_pred REAL,
            cls_score_vec_noise_pred REAL,
            cls_score_all_pred REAL,
            before_most_predictive_pred REAL,
            after_most_predictive_pred REAL,
            before_most_predictive_attn_pred REAL,
        )

    """)

    model_sums = {
        model_name: 0
        for model_name in model_names
    }
    total = len(dataset)

    for model_name in model_names:
        for category in dataset.get_imagenet_classes():
            os.makedirs(os.path.join(results_path, model_name, category), exist_ok=True)
            os.makedirs(os.path.join(bias_pl_resid_path, model_name, category), exist_ok=True)
            os.makedirs(os.path.join(bias_pl_vec_noise_path, model_name, category), exist_ok=True)
            os.makedirs(os.path.join(bias_pl_all_path, model_name, category), exist_ok=True)
            os.makedirs(os.path.join(cls_token_before_most_predictive_pred_path, 
                                     model_name, category), exist_ok=True)
            os.makedirs(os.path.join(cls_token_after_most_predictive_pred_path, 
                                     model_name, category), exist_ok=True)

    extractor_layers = {
        model_name: ['head']
        for model_name in model_names
    }
    bias_resid_hooks_handles = {
        model_name: []
        for model_name in model_names
    }

    residuals = {
        model_name: {}
        for model_name in model_names
    }

    def get_resid_hook(model: str, layer: str):
        def hook(module, input, output):
            residuals[model][layer] = input[0].detach()
        return hook

    for model_name, pred_inds in k_most_pred_by_model.items():
        # block_counts = pred_inds[:,0,:].flatten().bincount(minlength=len(models[model_name].blocks))
        block_counts_top1 = pred_inds[0,0,:].flatten().bincount(minlength=len(models[model_name].blocks))
        for i in range(len(models[model_name].blocks)):
            extractor_layers[model_name].append(f'blocks.{i}.mlp.fc1')
            extractor_layers[model_name].append(f'blocks.{i}.ls1')
            extractor_layers[model_name].append(f'blocks.{i}')
            # if block_counts[i].item() > 0:
            if block_counts_top1[i].item() > 0:
                handle = models[model_name].blocks[i].norm2.register_forward_hook(
                    get_resid_hook(model_name, f'blocks.{i}.norm2'))
                bias_resid_hooks_handles[model_name].append(handle)

    extractors = {
        model_name: create_feature_extractor(models[model_name], extractor_layers[model_name])
        for model_name in model_names
    }

    for i in tqdm(range(0, len(dataset), batch_size), desc='Batches'):
        
        length = min(len(dataset) - i, batch_size)
        items = [dataset[i+ii] for ii in range(length)]

        total += batch_size

        tensor_images_vanilla = torch.stack([item['img_vanilla'] for item in items], dim=0).to(device)
        tensor_images_miil = torch.stack([item['img_miil'] for item in items], dim=0).to(device)

        labels = torch.tensor([item['num_idx'] for item in items]).to(device)

        for model_name, extractor in extractors.items():

            extraction = extractor(tensor_images_vanilla if transform_keys[model_name] == 'vanilla' else tensor_images_miil)

            results = extraction['head']

            model_sums[model_name] += torch.gather(
                results, 1, labels.unsqueeze(1)).squeeze().sum().item()
            pred_vals, pred_inds = results.topk(k=k, dim=1)

            sums = results.sum(dim=1)
            l1_rows = results.abs().sum(dim=1)
            l2_rows = (results ** 2).sum(dim=1).sqrt()
            min_scores = results.min(dim=1)[0]
            exp_reciprocals = 1 / results.exp().sum(dim=1)

            head = model_heads[model_name]
            norm = model_norms[model_name]

            key_activations_batch = torch.stack([extraction[f'blocks.{i}.mlp.fc1'] 
                                           for i in range(len(models[model_name].blocks))],
                                           dim=0)
            block_cls_pred_batch = torch.stack([head(norm(extraction[f'blocks.{i}'][:,0,:]))
                                                for i in range(len(models[model_name].blocks))])
            attn_cls_pred_batch = torch.stack([head(norm(extraction[f'blocks.{i}.ls1'][:,0,:]))
                                               for i in range(len(models[model_name].blocks))])

            rows_pred = []
            rows_activations = []
            for ii, item in enumerate(items):

                result_path = os.path.join(results_path, model_name, item['imagenet_id'], Path(item['path']).stem) + '.pt'
                torch.save(results[ii].detach().cpu(), result_path)

                rows_pred.append((item['path'], model_name, item['imagenet_id'], 
                                  item['num_idx'], item['name'], sums[ii].item(), 
                                  l1_rows[ii].item(), l2_rows[ii].item(),
                                  exp_reciprocals[ii].item(), min_scores[ii].item(), result_path,
                                  *[pred_vals[ii, iii].item() for iii in range(k)],
                                  *[pred_inds[ii, iii].item() for iii in range(k)]))
                k_most_pred = k_most_pred_by_model[model_name]
                idx = item['num_idx']
                key_activations = key_activations_batch[:,ii,:,:]
                block_cls_pred = block_cls_pred_batch[:,ii,:]

                top_i_cls_token_act = key_activations[k_most_pred[:,0,idx], 0, k_most_pred[:,1,idx]]
                top_i_img_token_avg_act = key_activations[k_most_pred[:,0,idx], 1:, k_most_pred[:,1,idx]].mean(dim=1)

                _, _, hidden = key_activations.shape

                top_i_max_cls_token_act, top_i_max_cls_token_act_inds = (
                    key_activations[:,0,:].flatten().topk(k=k))
                
                # print(top_i_max_cls_token_act[:k])
                
                top_i_max_img_token_avg_act, top_i_max_img_token_avg_act_inds = (
                    key_activations[:,1:,:].mean(dim=1).flatten().topk(k=k))
                
                # print(top_i_max_img_token_avg_act[:k])

                top_i_max_cls_token_act_block_inds = top_i_max_cls_token_act_inds // hidden
                top_i_max_cls_token_act_col_inds = top_i_max_cls_token_act_inds % hidden

                top_i_max_img_token_avg_act_block_inds = top_i_max_img_token_avg_act_inds // hidden
                top_i_max_img_token_avg_act_col_inds = top_i_max_img_token_avg_act_inds % hidden

                fc2_top_weight = mlp_fc2_weights[model_name][k_most_pred[0,0,idx].item()]
                fc2_top_bias = mlp_fc2_biases[model_name][k_most_pred[0,0,idx].item()]
                top_bias_pred = mlp_fc2_biases_pred[model_name][k_most_pred[0,0,idx].item()]

                cls_residual = residuals[model_name][f'blocks.{k_most_pred[0,0,idx].item()}.norm2'][ii,0,:]
                vec_noise = F.gelu(torch.concat([key_activations[k_most_pred[0,0,idx], 0, :k_most_pred[0,1,idx]],
                                          key_activations[k_most_pred[0,0,idx], 0, k_most_pred[0,1,idx]+1:]],
                                          dim=-1)) @ torch.concat([fc2_top_weight[:k_most_pred[0,1,idx],:],
                                                                   fc2_top_weight[k_most_pred[0,1,idx]+1:]],
                                                                   dim=0)
                
                resid_pred = head(norm(cls_residual))
                top_bias_pl_res_l2 = cls_residual + fc2_top_bias
                top_bias_pl_res_pred = head(norm(top_bias_pl_res_l2))
                top_bias_pl_res_l2 = (top_bias_pl_res_l2 ** 2).sum().sqrt().item()

                vec_noise_pred = head(norm(vec_noise))
                top_bias_pl_vec_noise_l2 = vec_noise + fc2_top_bias
                top_bias_pl_vec_noise_pred = head(norm(top_bias_pl_vec_noise_l2))
                top_bias_pl_vec_noise_l2 = (top_bias_pl_vec_noise_l2 ** 2).sum().sqrt().item()

                all_pred = head(norm(vec_noise + cls_residual))
                top_bias_pl_all_l2 = cls_residual + vec_noise + fc2_top_bias
                top_bias_pl_all_pred = head(norm(top_bias_pl_all_l2))
                top_bias_pl_all_l2 = (top_bias_pl_all_l2 ** 2).sum().sqrt().item()

                before_most_predictive_pred = block_cls_pred[k_most_pred[0,0,idx].item()-1,:].clone()
                after_most_predictive_pred = block_cls_pred[k_most_pred[0,0,idx].item(),:].clone()
                before_most_predictive_attn_pred = attn_cls_pred_batch[k_most_pred[0,0,idx].item(),:].clone()

                top_bias_pl_res_pred_path = os.path.join(bias_pl_resid_path, model_name, item['imagenet_id'], Path(item['path']).stem) + '.pt'
                top_bias_pl_vec_noise_pred_path = os.path.join(bias_pl_vec_noise_path, model_name, item['imagenet_id'], Path(item['path']).stem) + '.pt'
                top_bias_pl_all_pred_path = os.path.join(bias_pl_all_path, model_name, item['imagenet_id'], Path(item['path']).stem) + '.pt'
                before_most_predictive_pred_path = os.path.join(cls_token_before_most_predictive_pred_path, model_name, item['imagenet_id'], Path(item['path']).stem) + '.pt'
                after_most_predictive_pred_path = os.path.join(cls_token_after_most_predictive_pred_path, model_name, item['imagenet_id'], Path(item['path']).stem) + '.pt'

                # if (((random_images[idx] * 1000) <= i+ii) & (i+ii < ((random_images[idx] + 1) * 1000))).any():
                torch.save(top_bias_pl_res_pred, top_bias_pl_res_pred_path)
                torch.save(top_bias_pl_vec_noise_pred, top_bias_pl_vec_noise_pred_path)
                torch.save(top_bias_pl_all_pred, top_bias_pl_all_pred_path)
                torch.save(before_most_predictive_pred, before_most_predictive_pred_path)
                torch.save(after_most_predictive_pred, after_most_predictive_pred_path)

                rows_activations.append((
                    item['path'], model_name, item['imagenet_id'], item['num_idx'], item['name'],
                    mlp_fc2_biases_l2[model_name][k_most_pred[0,0,idx].item()], top_bias_pl_res_l2,
                    top_bias_pl_vec_noise_l2, top_bias_pl_all_l2, top_bias_pl_res_pred_path,
                    top_bias_pl_vec_noise_pred_path, top_bias_pl_all_pred_path, top_bias_pl_res_pred.mean().item(),
                    top_bias_pl_vec_noise_pred.mean().item(), top_bias_pl_all_pred.mean().item(),
                    top_bias_pl_res_pred.std().item(), top_bias_pl_vec_noise_pred.std().item(),
                    top_bias_pl_all_pred.std().item(), top_bias_pred.mean().item(), top_bias_pred.std().item(),
                    resid_pred.mean().item(), resid_pred.std().item(), vec_noise_pred.mean().item(), 
                    vec_noise_pred.std().item(), all_pred.mean().item(), all_pred.std().item(),
                    top_bias_pl_res_pred.max().item(), top_bias_pl_vec_noise_pred.max().item(),
                    top_bias_pl_all_pred.max().item(), top_bias_pred.max().item(), resid_pred.max().item(),
                    vec_noise_pred.max().item(), all_pred.max().item(),
                    (cls_residual ** 2).sum().sqrt().item(),
                    (vec_noise ** 2).sum().sqrt().item(),
                    ((cls_residual + vec_noise) ** 2).sum().sqrt().item(),
                    top_bias_pl_res_pred[idx].item(), top_bias_pl_vec_noise_pred[idx].item(),
                    top_bias_pl_all_pred[idx].item(), top_bias_pred[idx].item(),
                    resid_pred[idx].item(), vec_noise_pred[idx].item(), all_pred[idx].item(),
                    before_most_predictive_pred[idx].item(), after_most_predictive_pred[idx].item(),
                    before_most_predictive_attn_pred[idx].item(),
                    *[top_i_cls_token_act[iii].item() for iii in range(k)],
                    *[top_i_img_token_avg_act[iii].item() for iii in range(k)],
                    *[k_most_pred[iii,0,idx].item() for iii in range(k)],
                    *[k_most_pred[iii,1,idx].item() for iii in range(k)],
                    *[top_i_max_cls_token_act[iii].item() for iii in range(k)],
                    *[top_i_max_cls_token_act_block_inds[iii].item() for iii in range(k)],
                    *[top_i_max_cls_token_act_col_inds[iii].item() for iii in range(k)],
                    *[top_i_max_img_token_avg_act[iii].item() for iii in range(k)],
                    *[top_i_max_img_token_avg_act_block_inds[iii].item() for iii in range(k)],
                    *[top_i_max_img_token_avg_act_col_inds[iii].item() for iii in range(k)]
                ))


            cursor.executemany(f"""
                    INSERT INTO vec_activations (path, pred_model, imagenet_id, num_idx, name, top_bias_l2,
                    top_bias_pl_res_l2, top_bias_pl_vec_noise_l2, top_bias_pl_all_l2, path_bias_pl_res_pred,
                    path_bias_pl_vec_noise_pred, path_bias_pl_all_pred, mean_bias_pl_res_pred, 
                    mean_bias_pl_vec_noise_pred, mean_bias_pl_all_pred, std_bias_pl_res_pred,
                    std_bias_pl_vec_noise_pred, std_bias_pl_all_pred, mean_top_bias_pred, std_top_bias_pred,
                    mean_resid_pred, std_resid_pred, mean_vec_noise_pred, std_vec_noise_pred,
                    mean_all_pred, std_all_pred, 
                    max_bias_pl_res_pred, max_bias_pl_vec_noise_pred, 
                    max_bias_pl_all_pred, max_top_bias_pred, max_resid_pred, max_vec_noise_pred, max_all_pred,
                    resid_l2, vec_noise_l2, all_l2,
                    cls_score_bias_pl_res_pred, cls_score_bias_pl_vec_noise_pred,
                    cls_score_bias_pl_all_pred, cls_score_top_bias_pred,
                    cls_score_resid_pred, cls_score_vec_noise_pred, cls_score_all_pred,
                    before_most_predictive_pred, after_most_predictive_pred,
                    before_most_predictive_attn_pred,
                    {','.join([f'top_{iii}_cls_token_act' for iii in range(k)])},
                    {','.join([f'top_{iii}_img_token_avg_act' for iii in range(k)])},
                    {','.join([f'top_{iii}_block_ind' for iii in range(k)])},
                    {','.join([f'top_{iii}_vec_ind' for iii in range(k)])},
                    {','.join([f'top_{iii}_max_cls_token_act' for iii in range(k)])},
                    {','.join([f'top_{iii}_max_cls_act_block_ind' for iii in range(k)])},
                    {','.join([f'top_{iii}_max_cls_act_vec_ind' for iii in range(k)])},
                    {','.join([f'top_{iii}_max_img_token_avg_act' for iii in range(k)])},
                    {','.join([f'top_{iii}_max_img_avg_act_block_ind' for iii in range(k)])},
                    {','.join([f'top_{iii}_max_img_avg_act_vec_ind' for iii in range(k)])}) VALUES
                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    {','.join(['?' for _ in range(10*k)])})
                """, rows_activations
            )
            cursor.executemany(f"""
                    INSERT INTO predictions (path, pred_model, imagenet_id, num_idx, name,
                        sum_row, l1_row, l2_row, exp_reciprocal, min_row, pred_path,
                        {",".join([f'top_{iii}_score' for iii in range(k)])},
                        {",".join([f'top_{iii}_ind' for iii in range(k)])}) VALUES 
                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, {", ".join(["?" for _ in range(2*k)])})
                """, rows_pred
            )
        connection.commit()
        for item in items:
            del item['img_miil'], item['img_vanilla']
        del tensor_images_miil, tensor_images_vanilla, labels, key_activations_batch, sums, l1_rows, l2_rows, before_most_predictive_pred, after_most_predictive_pred
        del min_scores, exp_reciprocals, block_cls_pred_batch, block_cls_pred
        del top_i_cls_token_act, top_i_img_token_avg_act, top_i_max_cls_token_act_inds, top_i_max_img_token_avg_act_inds
        del top_i_max_cls_token_act_block_inds, top_i_max_cls_token_act_col_inds, top_i_max_img_token_avg_act_block_inds
        del top_i_max_img_token_avg_act_col_inds, fc2_top_weight, fc2_top_bias, top_bias_pred, cls_residual, vec_noise, 
        del resid_pred, top_bias_pl_res_l2, top_bias_pl_res_pred, vec_noise_pred, 
        del top_bias_pl_vec_noise_l2, top_bias_pl_vec_noise_pred, all_pred, top_bias_pl_all_l2 
        del top_bias_pl_all_pred, top_bias_pl_res_pred_path, top_bias_pl_vec_noise_pred_path
        del top_bias_pl_all_pred_path
        torch.cuda.empty_cache()

    cursor.close()
    connection.close()

    for model_name in model_names:
        for handle in bias_resid_hooks_handles[model_name]:
            handle.remove()

    model_scores = {
        model_name: model_sums[model_name] / total
        for model_name in model_names
    }

path = 'notebooks\\imgnet_val.db'
results_path = 'A:\\CVData\\ImageNet\\val_img_pred'
bias_pl_resid_path = 'A:\\CVData\\ImageNet\\val_top_bias_pl_resid_pred'
bias_pl_vec_noise_path = 'A:\\CVData\\ImageNet\\val_top_bias_pl_vec_noise_pred'
bias_pl_all_path = 'A:\\CVData\\ImageNet\\val_top_bias_pl_all_pred'
cls_token_before_most_predictive_pred = 'A:\\CVData\\ImageNet\\val_cls_token_before_most_predictive_pred'
cls_token_after_most_predictive_pred = 'A:\\CVData\\ImageNet\\val_cls_token_after_most_predictive_pred'
# print('imagenet val eval')
# create_pred_db(path, results_path, bias_pl_resid_path, bias_pl_vec_noise_path, bias_pl_all_path,
#                cls_token_before_most_predictive_pred, cls_token_after_most_predictive_pred)

dataset_top1_clear_base = MixedPredictiveImagesDataset(
    'A:\\CVData\\project-images\\images-mixed-224-fc1-multimodel', tops={'1'}, gen_types='clear',
    models={'vit_base_patch16_224'})
dataset_top1_clear_basemiil = MixedPredictiveImagesDataset(
    'A:\\CVData\\project-images\\images-mixed-224-fc1-multimodel', tops={'1'}, gen_types='clear',
    models={'vit_base_patch16_224_miil'})
dataset_top1_clear_base32 = MixedPredictiveImagesDataset(
    'A:\\CVData\\project-images\\images-mixed-224-fc1-multimodel', tops={'1'}, gen_types='clear',
    models={'vit_base_patch32_224'})
dataset_top1_clear_large = MixedPredictiveImagesDataset(
    'A:\\CVData\\project-images\\images-mixed-224-fc1-multimodel', tops={'1'}, gen_types='clear',
    models={'vit_large_patch16_224'})
dataset_top1_clear_base_2 = MixedPredictiveImagesDataset(
    'A:\\CVData\\project-images\\images-mixed-224-fc1-multimodel-2', tops={'1'}, gen_types='clear',
    models={'vit_base_patch16_224'})
dataset_top1_clear_basemiil_2 = MixedPredictiveImagesDataset(
    'A:\\CVData\\project-images\\images-mixed-224-fc1-multimodel-2', tops={'1'}, gen_types='clear',
    models={'vit_base_patch16_224_miil'})
dataset_top1_clear_base32_2 = MixedPredictiveImagesDataset(
    'A:\\CVData\\project-images\\images-mixed-224-fc1-multimodel-2', tops={'1'}, gen_types='clear',
    models={'vit_base_patch32_224'})
dataset_top1_clear_large_2 = MixedPredictiveImagesDataset(
    'A:\\CVData\\project-images\\images-mixed-224-fc1-multimodel-2', tops={'1'}, gen_types='clear',
    models={'vit_large_patch16_224'})


m_dataset_top1_base = MixedPredictiveImagesMacoDataset(
    'A:\\CVData\\project-images\\images-mixed-1280-fc1-multimodel-maco', 
    tops={'1'}, models={'vit_base_patch16_224'})
m_dataset_top1_basemiil = MixedPredictiveImagesMacoDataset(
    'A:\\CVData\\project-images\\images-mixed-1280-fc1-multimodel-maco', 
    tops={'1'}, models={'vit_base_patch16_224_miil'})
m_dataset_top1_base32 = MixedPredictiveImagesMacoDataset(
    'A:\\CVData\\project-images\\images-mixed-1280-fc1-multimodel-maco', 
    tops={'1'}, models={'vit_base_patch32_224'})
m_dataset_top1_large = MixedPredictiveImagesMacoDataset(
    'A:\\CVData\\project-images\\images-mixed-1280-fc1-multimodel-maco', 
    tops={'1'}, models={'vit_large_patch16_224'})
m_dataset_top1_base_2 = MixedPredictiveImagesMacoDataset(
    'A:\\CVData\\project-images\\images-mixed-1280-fc1-multimodel-maco-2', 
    tops={'1'}, models={'vit_base_patch16_224'})
m_dataset_top1_basemiil_2 = MixedPredictiveImagesMacoDataset(
    'A:\\CVData\\project-images\\images-mixed-1280-fc1-multimodel-maco-2', 
    tops={'1'}, models={'vit_base_patch16_224_miil'})
m_dataset_top1_base32_2 = MixedPredictiveImagesMacoDataset(
    'A:\\CVData\\project-images\\images-mixed-1280-fc1-multimodel-maco-2', 
    tops={'1'}, models={'vit_base_patch32_224'})
m_dataset_top1_large_2 = MixedPredictiveImagesMacoDataset(
    'A:\\CVData\\project-images\\images-mixed-1280-fc1-multimodel-maco-2', 
    tops={'1'}, models={'vit_large_patch16_224'})
# dataset_top12_uw_clear = MixedPredictiveImagesDataset(
#     'A:\\CVData\\project-images\\images-mixed-224-fc1-multimodel', tops={'1,2'}, gen_types='clear', weighting_scheme='unweighted')
# dataset_top12_sm_clear = MixedPredictiveImagesDataset(
#     'A:\\CVData\\project-images\\images-mixed-224-fc1-multimodel', tops={'1,2'}, gen_types='clear', weighting_scheme='softmax')
# dataset_top123_uw_clear = MixedPredictiveImagesDataset(
#     'A:\\CVData\\project-images\\images-mixed-224-fc1-multimodel', tops={'1,2,3'}, gen_types='clear', weighting_scheme='unweighted')
# dataset_top123_sm_clear = MixedPredictiveImagesDataset(
#     'A:\\CVData\\project-images\\images-mixed-224-fc1-multimodel', tops={'1,2,3'}, gen_types='clear', weighting_scheme='softmax')
# dataset_top1234_uw_clear = MixedPredictiveImagesDataset(
#     'A:\\CVData\\project-images\\images-mixed-224-fc1-multimodel', tops={'1,2,3,4'}, gen_types='clear',
#     weighting_scheme='unweighted'
# )
# dataset_top1234_sm_clear = MixedPredictiveImagesDataset(
#     'A:\\CVData\\project-images\\images-mixed-224-fc1-multimodel', tops={'1,2,3,4'}, gen_types='clear',
#     weighting_scheme='softmax'
# )
# dataset_top1_div = MixedPredictiveImagesDataset(
#     'A:\\CVData\\project-images\\images-mixed-224-fc1-multimodel', tops={'1'}, gen_types='div')
# dataset_top12_uw_div = MixedPredictiveImagesDataset(
#     'A:\\CVData\\project-images\\images-mixed-224-fc1-multimodel', tops={'1,2'}, gen_types='div', weighting_scheme='unweighted')
# dataset_top12_sm_div = MixedPredictiveImagesDataset(
#     'A:\\CVData\\project-images\\images-mixed-224-fc1-multimodel', tops={'1,2'}, gen_types='div', weighting_scheme='softmax')
# dataset_top123_uw_div = MixedPredictiveImagesDataset(
#     'A:\\CVData\\project-images\\images-mixed-224-fc1-multimodel', tops={'1,2,3'}, gen_types='div', weighting_scheme='unweighted')
# dataset_top123_sm_div = MixedPredictiveImagesDataset(
#     'A:\\CVData\\project-images\\images-mixed-224-fc1-multimodel', tops={'1,2,3'}, gen_types='div', weighting_scheme='softmax')
# dataset_top1234_uw_div = MixedPredictiveImagesDataset(
#     'A:\\CVData\\project-images\\images-mixed-224-fc1-multimodel', tops={'1,2,3,4'}, gen_types='div',
#     weighting_scheme='unweighted'
# )
# dataset_top1234_sm_div = MixedPredictiveImagesDataset(
#     'A:\\CVData\\project-images\\images-mixed-224-fc1-multimodel', tops={'1,2,3,4'}, gen_types='div',
#     weighting_scheme='softmax'
# )

# print("fid top_1_clear base")
# top_1_clear_base = fid_score([dataset_top1_clear_base, dataset_top1_clear_base_2], 
#                              'A:\\CVData\\ImageNet\\val_fid_distr',
#                         store_path='A:\\CVData\\project-images\\images-mixed-224-fc1-multimodel\\fid_top1_clear_vit_base_patch16_224.json',
#                         device=device)
# print("fid top_1_clear basemiil")
# top_1_clear_base = fid_score([dataset_top1_clear_basemiil, dataset_top1_clear_basemiil_2], 
#                              'A:\\CVData\\ImageNet\\val_fid_distr',
#                         store_path='A:\\CVData\\project-images\\images-mixed-224-fc1-multimodel\\fid_top1_clear_vit_base_patch16_224_miil.json',
#                         device=device)
# print("fid top_1_clear base32")
# top_1_clear_base = fid_score([dataset_top1_clear_base32, dataset_top1_clear_base32_2], 
#                              'A:\\CVData\\ImageNet\\val_fid_distr',
#                         store_path='A:\\CVData\\project-images\\images-mixed-224-fc1-multimodel\\fid_top1_clear_vit_base_patch32_224.json',
#                         device=device)
# print("fid top_1_clear large")
# top_1_clear_base = fid_score([dataset_top1_clear_large, dataset_top1_clear_large_2], 
#                              'A:\\CVData\\ImageNet\\val_fid_distr',
#                         store_path='A:\\CVData\\project-images\\images-mixed-224-fc1-multimodel\\fid_top1_clear_vit_large_patch16_224.json',
#                         device=device)

# print('fid top_1 maco base')
# top_1 = fid_score([m_dataset_top1_base, m_dataset_top1_base_2], 
#                   'A:\\CVData\\ImageNet\\val_fid_distr',
#                         store_path='A:\\CVData\\project-images\\images-mixed-1280-fc1-multimodel-maco\\fid_top1_vit_base_patch16_224.json',
#                         device=device)
# print('fid top_1 maco basemiil')
# top_1 = fid_score([m_dataset_top1_basemiil, m_dataset_top1_basemiil_2], 
#                   'A:\\CVData\\ImageNet\\val_fid_distr',
#                         store_path='A:\\CVData\\project-images\\images-mixed-1280-fc1-multimodel-maco\\fid_top1_vit_base_patch16_224_miil.json',
#                         device=device)
# print('fid top_1 maco base32')
# top_1 = fid_score([m_dataset_top1_base32, m_dataset_top1_base32_2], 
#                   'A:\\CVData\\ImageNet\\val_fid_distr',
#                         store_path='A:\\CVData\\project-images\\images-mixed-1280-fc1-multimodel-maco\\fid_top1_vit_base_patch32_224.json',
#                         device=device)
# print('fid top_1 maco large')
# top_1 = fid_score([m_dataset_top1_large, m_dataset_top1_large_2], 
#                   'A:\\CVData\\ImageNet\\val_fid_distr',
#                         store_path='A:\\CVData\\project-images\\images-mixed-1280-fc1-multimodel-maco\\fid_top1_vit_large_patch16_224.json',
#                         device=device)

# print("fid top_12_uw_clear")
# top_12_uw_clear = fid_score(dataset_top12_uw_clear, 'A:\\CVData\\ImageNet\\val_fid_distr',
#                             store_path='A:\\CVData\\project-images\\images-mixed-224-fc1-multimodel\\fid_top12_uw_clear_corrected.json',
#                             device=device)
# print('fid top_12_sm_clear')
# top_12_sm_clear = fid_score(dataset_top12_sm_clear, 'A:\\CVData\\ImageNet\\val_fid_distr',
#                             store_path='A:\\CVData\\project-images\\images-mixed-224-fc1-multimodel\\fid_top12_sm_clear_corrected.json',
#                             device=device)
# print('fid top_123_uw_clear')
# top_123_uw_clear = fid_score(dataset_top123_uw_clear, 'A:\\CVData\\ImageNet\\val_fid_distr',
#                              store_path='A:\\CVData\\project-images\\images-mixed-224-fc1-multimodel\\fid_top123_uw_clear_corrected.json',
#                              device=device)
# print('fid top_123_sm_clear')
# top_123_sm_clear = fid_score(dataset_top123_sm_clear, 'A:\\CVData\\ImageNet\\val_fid_distr',
#                              store_path='A:\\CVData\\project-images\\images-mixed-224-fc1-multimodel\\fid_top123_sm_clear_corrected.json',
#                              device=device)
# print('fid top_1234_uw_clear')
# top_1234_uw_clear = fid_score(dataset_top1234_uw_clear, 'A:\\CVData\\ImageNet\\val_fid_distr',
#                               store_path='A:\\CVData\\project-images\\images-mixed-224-fc1-multimodel\\fid_top1234_uw_clear_corrected.json',
#                               device=device)
# print('fid top_1234_sm_clear')
# top_1234_sm_clear = fid_score(dataset_top1234_sm_clear, 'A:\\CVData\\ImageNet\\val_fid_distr',
#                               store_path='A:\\CVData\\project-images\\images-mixed-224-fc1-multimodel\\fid_top1234_sm_clear_corrected.json',
#                               device=device)
# print('fid top_1_div')
# top_1_div = fid_score(dataset_top1_div, 'A:\\CVData\\ImageNet\\val_fid_distr',
#                         store_path='A:\\CVData\\project-images\\images-mixed-224-fc1-multimodel\\fid_top1_div_corrected.json',
#                         device=device)
# print('fid top_12_uw_div')
# top_12_uw_div = fid_score(dataset_top12_uw_div, 'A:\\CVData\\ImageNet\\val_fid_distr',
#                             store_path='A:\\CVData\\project-images\\images-mixed-224-fc1-multimodel\\fid_top12_uw_div_corrected.json',
#                             device=device)
# print('fid top_12_sm_div')
# top_12_sm_div = fid_score(dataset_top12_sm_div, 'A:\\CVData\\ImageNet\\val_fid_distr',
#                             store_path='A:\\CVData\\project-images\\images-mixed-224-fc1-multimodel\\fid_top12_sm_div_corrected.json',
#                             device=device)
# print('fid top_123_uw_div')
# top_123_uw_div = fid_score(dataset_top123_uw_div, 'A:\\CVData\\ImageNet\\val_fid_distr',
#                              store_path='A:\\CVData\\project-images\\images-mixed-224-fc1-multimodel\\fid_top123_uw_div_corrected.json',
#                              device=device)
# print('fid top_123_sm_div')
# top_123_sm_div = fid_score(dataset_top123_sm_div, 'A:\\CVData\\ImageNet\\val_fid_distr',
#                              store_path='A:\\CVData\\project-images\\images-mixed-224-fc1-multimodel\\fid_top123_sm_div_corrected.json',
#                              device=device)
# print('fid top_1234_uw_div')
# top_1234_uw_div = fid_score(dataset_top1234_uw_div, 'A:\\CVData\\ImageNet\\val_fid_distr',
#                               store_path='A:\\CVData\\project-images\\images-mixed-224-fc1-multimodel\\fid_top1234_uw_div_corrected.json',
#                               device=device)
# print('fid top_1234_sm_div')
# top_1234_sm_div = fid_score(dataset_top1234_sm_div, 'A:\\CVData\\ImageNet\\val_fid_distr',
#                               store_path='A:\\CVData\\project-images\\images-mixed-224-fc1-multimodel\\fid_top1234_sm_div_corrected.json',
#                               device=device)

# dataset_top1 = MixedPredictiveImagesMacoDataset(
#     'A:\\CVData\\project-images\\images-mixed-1280-fc1-multimodel-maco', 
#     tops={'1'})
# dataset_top1234_uw = MixedPredictiveImagesMacoDataset(
#     'A:\\CVData\\project-images\\images-mixed-1280-fc1-multimodel-maco', 
#     tops={'1,2,3,4'}, weighting_scheme='unweighted')
# dataset_top1234_sm = MixedPredictiveImagesMacoDataset(
#     'A:\\CVData\\project-images\\images-mixed-1280-fc1-multimodel-maco', 
#     tops={'1,2,3,4'}, weighting_scheme='softmax')

# print('fid top_1')
# top_1 = fid_score(dataset_top1, 'A:\\CVData\\ImageNet\\val_fid_distr',
#                         store_path='A:\\CVData\\project-images\\images-mixed-1280-fc1-multimodel-maco\\fid_top1_corrected.json',
#                         device=device)
# print('fid top_1234_uw')
# top_1234_uw = fid_score(dataset_top1234_uw, 'A:\\CVData\\ImageNet\\val_fid_distr',
#                               store_path='A:\\CVData\\project-images\\images-mixed-1280-fc1-multimodel-maco\\fid_top1234_uw_corrected.json',
#                               device=device)
# print('fid top_1234_sm')
# top_1234_sm = fid_score(dataset_top1234_sm, 'A:\\CVData\\ImageNet\\val_fid_distr',
#                               store_path='A:\\CVData\\project-images\\images-mixed-1280-fc1-multimodel-maco\\fid_top1234_sm_corrected.json',
#                               device=device)


datasets_by_block = {
    (i, gen_type): ImagesByBlockDataset('A:\\CVData\\project-images\\images-by-block-224-fc1',
                            models=['vit_base_patch16_224', 'vit_base_patch32_224'],
                            block_by_model={'vit_base_patch16_224': [i], 
                                             'vit_base_patch32_224': [i]},
                            gen_types=gen_type)
    for i in range(5, 12) for gen_type in ['clear', 'cls_token']
}

for (block, gen_type), ds in datasets_by_block.items():
    print(f'fid by block {block} and gen_type {gen_type}')
    fid_score(ds, 'A:\\CVData\\ImageNet\\val_fid_distr',
              store_path=f'A:\\CVData\\project-images\\images-by-block-224-fc1\\block_{block}_{gen_type}.json', device=device)

# print('plausibility by block')
# plausibility = plausibility_score(imagenet, ['A:\\CVData\\project-images\\images-by-block-224-fc1\\plausibility_with_gen.db'], [dataset_by_block], device=device)

