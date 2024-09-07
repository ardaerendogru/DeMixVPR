import torch
import torchvision.transforms as T
from PIL import Image

def denormalize(tensor):
    """
    Denormalize a torch tensor that was normalized with the given mean and std,
    and convert it to a PIL image.
    
    Args:
        tensor (torch.Tensor): Normalized image tensor
    
    Returns:
        PIL.Image: Denormalized image as a PIL Image
    """
    # Ensure input is a torch tensor
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a torch tensor")

    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(tensor.device)
    denormalized_tensor = tensor * std + mean

    # Clamp values to be between 0 and 1
    denormalized_tensor = torch.clamp(denormalized_tensor, 0, 1)

    return denormalized_tensor
