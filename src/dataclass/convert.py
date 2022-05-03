# Credits : Sagar Vinodababu
# This code was adapted from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution
from PIL import Image
import torchvision.transforms.functional as FT
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def convert_image(image, source, target):
    """Convert an image from a source format to a target format.

    Args:
        image (PIL image): The image to convert
        source (str): "pil" || "[0, 1]" || "[-1, 1]"
        target (str): "pil" || "[0, 1]" || "[-1, 1]" || "[0, 255]"
                      || "imagenet-normalized"

    Returns:
        converted image
    """

    # Convert from source to [0, 1]
    if source == 'pil':
        image = FT.to_tensor(image)

    elif source == '[0, 1]':
        pass  # already in [0, 1]

    elif source == '[-1, 1]':
        image = (image + 1.) / 2.

    # Convert from [0, 1] to target
    if target == 'pil':
        image = FT.to_pil_image(image)

    elif target == '[0, 255]':
        image = 255. * image

    elif target == '[0, 1]':
        pass  # already in [0, 1]

    elif target == '[-1, 1]':
        image = 2. * image - 1.

    elif target == 'imagenet-normalized':
        """
        see. https://pytorch.org/vision/stable/models.html
        Convert into this format before entering the model.
        """
        imagenet_mean = torch.FloatTensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
        imagenet_std = torch.FloatTensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)
        imagenet_mean_cuda = torch.FloatTensor([0.485, 0.456, 0.406]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        imagenet_std_cuda = torch.FloatTensor([0.229, 0.224, 0.225]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        if image.ndimension() == 3:
            image = (image - imagenet_mean) / imagenet_std
        elif image.ndimension() == 4:
            image = (image - imagenet_mean_cuda) / imagenet_std_cuda

    return image
