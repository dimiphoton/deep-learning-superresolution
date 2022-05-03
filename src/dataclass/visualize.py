#!/usr/bin/env python3

import torch

from dataset import ImagesDataset

if __name__ == '__main__':
    test_dataset = ImagesDataset("./", 400, 2, 'imagenet-norm', "[-1, 1]", True)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, pin_memory=True)

    for i, (lr_images, hr_images) in enumerate(test_loader):
        print(type(lr_images))
        print(type(hr_images))
