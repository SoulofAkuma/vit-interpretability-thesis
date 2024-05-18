from typing import Union, List
import torch
from timm.models.vision_transformer import VisionTransformer
from transformers import AutoImageProcessor
import timm.data
import PIL.Image
from typing import Dict, Callable, Tuple

PREPROCESSOR_MAP: Dict[str, Callable[[VisionTransformer], Callable[[PIL.Image.Image], torch.Tensor]]] = {
    'vit_base_patch32_224': lambda model: AutoImageProcessor.from_pretrained('google/vit-base-patch32-224-in21k'),
    'vit_base_patch16_224': lambda model: AutoImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k'),
    'vit_large_patch16_224': lambda model: AutoImageProcessor.from_pretrained('google/vit-large-patch16-224-in21k'),
    'vit_base_patch16_224_miil': lambda model: timm.data.create_transform(**timm.data.resolve_data_config({}, model=model), isTraining=False)  
}

def embedding_projection(vit: VisionTransformer, values: Union[List[torch.Tensor],torch.Tensor], device=None) -> torch.Tensor:
    """Project the value vectors onto the class embedding space of the transformer

    Args:
        vit (VisionTransformer): The vision transformer
        values (Union[List[torch.Tensor],torch.Tensor]): The list of value vector matrices or a 3 dimensional value vectors matrix of shape (number of value vectors, hidden_dim)

    Returns:
        torch.Tensor: The projection of each of the value vector matrices
    """
    proj = vit.head.eval()
    norm = vit.norm.eval()

    if device is not None:
        proj = proj.to(device)
        norm = norm.to(device)

    if type(values) is list:
        values = torch.stack(values, dim=0)

    if device is not None:
        values = values.to(device)

    result = proj(norm(values)).detach()
    return result if device is None else result.to(device)

def load_model_with_preprocessor(model_descriptor: str) -> Tuple[VisionTransformer, Callable[[PIL.Image.Image], torch.Tensor]]:
    model = timm.create_model(model_descriptor)
    transform = PREPROCESSOR_MAP[model_descriptor](model)

    return model, transform