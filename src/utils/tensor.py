import torch

# adapted from the maco implementation https://github.com/serre-lab/Horama
def standardize(tensor: torch.Tensor) -> torch.Tensor:
    """Standardizes a tensor to have 0 mean and 1 std deviation.

    Args:
        tensor (torch.Tensor): The tensor to standardize.

    Returns:
        torch.Tensor: The standardized tensor.
    """
    tensor = tensor - torch.mean(tensor)
    tensor = tensor / (torch.std(tensor) + 1e-4)
    return tensor

# adapted from the maco implementation https://github.com/serre-lab/Horama
def recorrelate_colors(image: torch.Tensor, device: str) -> torch.Tensor:
    """Standardize the colors in an RGB image of shape (3, H, W) 

    Args:
        image (torch.Tensor): The image of shape (3, H, W).
        device (str): The device to move the recorrelated image to.

    Returns:
        torch.Tensor: The image with recorrelated colors on the specified device.
    """

    assert len(image.shape) == 3

    # tensor for color correlation svd square root
    color_correlation_svd_sqrt = torch.tensor(
        [[0.56282854, 0.58447580, 0.58447580],
         [0.19482528, 0.00000000, -0.19482528],
         [0.04329450, -0.10823626, 0.06494176]],
        dtype=torch.float32
    ).to(device)

    permuted_image = image.permute(1, 2, 0).contiguous()
    flat_image = permuted_image.view(-1, 3)

    recorrelated_image = torch.matmul(flat_image, color_correlation_svd_sqrt)
    recorrelated_image = recorrelated_image.view(permuted_image.shape).permute(2, 0, 1)

    return recorrelated_image

def clip_quantile(image: torch.Tensor, quantile: float) -> torch.Tensor:
    """Clip the values in the tensor across all dimensions to the range [quantile, 1-quantile] \
    calculated across all dimesions.

    Args:
        image (torch.Tensor): The image of shape (C, H, W) to clip .
        quantile (float): The quantile value between 0 and 1.

    Returns:
        torch.Tensor: The clipped image of the same shape as the input.
    """
    return torch.clip(image, torch.quantile(image, quantile), torch.quantile(image, 1 - quantile))

def normalize(image):
    """Min-Max normalize the image across all dimensions by subtracting the minimum value over all \
    dimensions and dividing by the now maximum value across all dimensions.

    Args:
        image (torch.Tensor): The image to normalize of shape (C, H, W)

    Returns:
        torch.Tensor: The normalized image of the same shape as the input
    """
    image -= image.min()
    image /= image.max()
    return image