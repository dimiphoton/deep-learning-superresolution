# Credits : Sagar Vinodababu
# This code was adapted from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution
from .convert import convert_image
import os
from PIL import Image
import json
from torch.utils.data import Dataset
import random
import math


class ImagesDataset(Dataset):

    def __init__(self, data_folder, crop_size, scaling_factor, hr_format, lr_format, train=True):
        """
        Params
            data_folder(str): The folder where are the JSON files containing the path to images.
            crop_size (int): The crop size with which to crop HR images.
            scaling_factor (int): The scaling factor with which to downscale the HR 
                                image to obtain the LR image.
            hr_format (str): The format that will have the HR image.
            lr_format (str): The format that will have the LR image.
            train (bool): True if it is the train set, False if it is the test one.


        N.B. Available formats : ["pil" || "[0, 1]" || "[-1, 1]" || "[0, 255]" 
                                        || "imagenet-normalized"]
        """
        self.data_folder = data_folder
        self.crop_size = crop_size
        self.scaling_factor = scaling_factor
        self.hr_format = hr_format
        self.lr_format = lr_format
        self.train = train

        # The crop dimensions must be divisible by the scaling-factor
        assert crop_size % scaling_factor == 0, "Crop dimensions are not perfectly divisible by scaling factor!"


        # Read Data from pre-created JSON
        if train:
            with open(os.path.join(data_folder, 'train_images.json'), 'r') as f:
                self.images = json.load(f)
        else:
            with open(os.path.join(data_folder, 'test_images.json'), 'r') as f:
                self.images = json.load(f)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Read each image
        image = Image.open(self.images[index], mode='r')
        image = image.convert("RGB")

        # Create LR images
        hr_image, lr_image = create_HRLR(image, self.crop_size, self.scaling_factor,
                                        self.hr_format, self.lr_format, self.train)

        return lr_image, hr_image


def create_HRLR(image, crop_size, scaling_factor, hr_format, lr_format, train):
    """ Create the HR & LR images (cropped) and downscaled (LR)

    Params
        image (PIL): The image to crop and downscale
        crop_size (int): The crop size with which to crop HR images.
        scaling_factor (int): The scaling factor with which to downscale the HR 
                            image to obtain the LR image.
        hr_format (str): The format that will have the HR image.
        lr_format (str): The format that will have the LR image.
        train (bool): True if it is the train set, False if it is the test one.

    N.B. Available formats : ["pil" || "[0, 1]" || "[-1, 1]" || "[0, 255]" 
                                    || "imagenet-normalized"]

    Returns
        hr_image, lr_image: A pair of the HR & LR image in the given format
    """

    # Train dataset : We crop the HR image randomly with a fixed size.
    # (The cropping size is divisible by the scaling factor (verified before))
    if train:
        left = random.randint(1, image.width - crop_size)
        top = random.randint(1, image.height - crop_size)
    else:
        left = (image.width - crop_size) // 2
        top = (image.height - crop_size) // 2

    right = left + crop_size
    bottom = top + crop_size
    hr_image = image.crop((left, top, right, bottom))


    # Downscale the HR to obtain the LR image
    lr_image = hr_image.resize((int(hr_image.width / scaling_factor), int(hr_image.height / scaling_factor)),
                        Image.BICUBIC)

    # Convert into the wanted type
    lr_image = convert_image(lr_image, source='pil', target=lr_format)
    hr_image = convert_image(hr_image, source='pil', target=hr_format)

    return hr_image, lr_image
