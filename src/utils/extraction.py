import torch
from typing import List, Optional, Union
from timm.models.vision_transformer import VisionTransformer
from torchvision.models.feature_extraction import create_feature_extractor

__KEY_VECTOR_EXTRACTORS = {}

def extract_value_vectors(vit: VisionTransformer, device: Optional[str]=None, with_bias: bool=False, 
                          value_weight_with_bias: float=1.0, stack: bool=False) -> List[torch.Tensor]:
    """Extract the value vectors of the MLP heads of a vision transformer by block

    Args:
        vit (nn.Module): A vision transformer
        device (str, optional): The device to move the result to. Defaults to None.
        with_bias (bool, optional): Whether to add the value bias to each projected value vector. Defaults to False.
        value_weight_with_bias (float, optional): The weight for the value vector, when adding the bias to it. Defaults to 1.0.
    Returns:
        List[torch.Tensor]: A list of value vector matrices for each block
    """
    result = [block.mlp.fc2.weight.detach().T for block in vit.blocks]
    if with_bias:
        result = [value_weight_with_bias * result[i] + block.mlp.fc2.bias.detach() 
                  for i, block in enumerate(vit.blocks)]
        
    result = result if device is None else [t.to(device) for t in result]

    return torch.stack(result, dim=0) if stack else result

def extract_value_biases(vit: VisionTransformer, device: Optional[str]=None,
                         stack: bool=False) -> Union[List[torch.Tensor], torch.Tensor]:
    """Extract the biases of all MLP value vector matrices in the ViT.

    Args:
        vit (VisionTransformer): The ViT to extract from
        device (Optional[str], optional): The device to move the biases to. Defaults to None.
        concat (bool, optional): Whether to concat the biases into one tensor. Defaults to False.

    Returns:
        Union[List[torch.Tensor], torch.Tensor]: A list of biases or the biases concatenated as one tensor.
    """
    result = [block.mlp.fc2.bias.detach() for block in vit.blocks]
    result = result if device is None else [bias.to(device) for bias in result]
    return torch.stack(result, dim=0) if stack else result


def extract_mhsa_proj_vectors(vit: VisionTransformer, device=None) -> List[torch.Tensor]:
    """Extract the projection vectors of the MHSA blocks of a vision transformer by block

    Args:
        vit (VisionTransformer): a vision transformer 
        device (string, optional): The device to move the result to. Defaults to None.

    Returns:
        List[torch.Tensor]: A list of projection vector matrices for each block
    """
    result = [block.attn.proj.weight.detach().T for block in vit.blocks]
    return result if device is None else [t.to(device) for t in result]

def extract_mhsa_value_vectors(vit: VisionTransformer, embed_dim: int=768,
                               device=None) -> List[torch.Tensor]:
    """Extract the value vectors of the MHSA blocks of a vision transformer by block

    Args:
        vit (VisionTransformer): A vision transformer
        embed_dim (int, optional): The embedding dim of the vision transformer to extract
        the correct slice of the qkv weight matrix. Defaults to 768.
        device (_type_, optional): The device to move the result to. Defaults to None.

    Returns:
        List[torch.Tensor]: A list of value vector matrices for each block
    """
    result = [block.attn.qkv.weight.detach().T[:,2*embed_dim:] for block in vit.blocks]
    return result if device is None else [t.to(device) for t in result]

def extract_key_weights(vit: VisionTransformer) -> List[torch.Tensor]:
    """Extract the key vectors of the MLP heads of a vision transformer by block

    Args:
        vit (nn.Module): A vision transformer

    Returns:
        List[torch.Tensor]: A list of key vector matrices for each block
    """
    return [block.mlp.fc1.weight.detach().T for block in vit.blocks]

def extract_computed_key_vectors(vit: VisionTransformer, 
                                 images: Union[List[torch.Tensor], torch.Tensor],
                                 device: str=None) -> torch.Tensor:
    """Extract the intermediate results of the input data and the projected key vectors 
        after going through the activation function i.e. GELU(X @ W_key) for all blocks

    Args:
        vit (VisionTransformer): A vision transformer
        images (Union[List[torch.Tensor], torch.Tensor]): The preprocessed images to calculate the
            intermediate values for. If this is a list, it can either consist of individual images 
            or batches of images and will be converted to a large batch of images internally. If 
            this is a tensor, it can either be an individual image or a batch of images.
        device (str, optional): The device to 

    Returns:
        torch.Tensor: The intermediate results with the shape  (nr of blocks, nr of images 
            nr of patches, nr of value vectors/hidden dim)
    """

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if type(images) is list:
        images = torch.concat([image.unsqueeze(0) if len(image.shape) == 3 else image for image in images]).to(device)
    if len(images.shape) == 3:
        images = images.unsqueeze(0).to(device)

    extract_layers = [f'blocks.{i}.mlp.act' for i in range(len(vit.blocks))]
    if (vit, device) not in __KEY_VECTOR_EXTRACTORS:
        __KEY_VECTOR_EXTRACTORS[(vit, device)] = create_feature_extractor(vit, extract_layers).to(device)

    results = __KEY_VECTOR_EXTRACTORS[(vit, device)](images)
    return torch.stack([results[layer] for layer in extract_layers], dim=0).to(device)