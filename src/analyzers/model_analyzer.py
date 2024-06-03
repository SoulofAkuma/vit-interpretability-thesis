import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from src.utils.IndexDataset import IndexDataset
from typing import Tuple
import tqdm
from src.utils.transformation import transform_images

def correct_prediction_rate(model: nn.Module, dataset: IndexDataset,
                            huggingface_model_descriptor: str, device=None, batch_size: int=10,
                            show_progression: bool=True) -> Tuple[float, torch.Tensor]:
    
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    predicted_logits = torch.empty(len(dataset)).to(device)
    correct_logits = torch.empty(len(dataset)).to(device)

    loop = range(0, len(dataset), batch_size)
    if show_progression:
        loop = tqdm.tqdm(loop)

    for i in loop:
        range_size = min(batch_size, len(dataset) - i)
        imgs = [dataset[ii] for ii in range(i, i+range_size)]
        tensor_imgs = torch.concat(transform_images([imgs[ii]['img'] for ii in range(len(imgs))], 
                                                    huggingface_model_descriptor, device), dim=0)

        correct_logits[i:i+range_size] = torch.tensor([imgs[ii]['num_idx'] for ii in range(len(imgs))])

        prediction = model(tensor_imgs)
        prediction = prediction.argmax(dim=1)

        predicted_logits[i:i+range_size] = prediction

    return (predicted_logits == correct_logits).mean(dtype=torch.float32), predicted_logits