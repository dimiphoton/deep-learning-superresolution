#!/usr/bin/env python3
# Credits : Sagar Vinodababu
# This code was adapted from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution

import os
import json
from PIL import Image

"""
Creates a JSON list with all the train & test data needed with a minimum size 
so that we are sure we can crop them. It allows us to use several datasets and 
thus increase the number of images for the training.
"""

# If we will create the LR images ourselves
def prepare_datasets(train_folders, test_folders, min_size, output_folder):
    """Creates 2 JSON files containing the path for the train and test images.
       (1. train_images.json
        2. test_images.json)

    Args:
        train_folders (str): folders containing the training images.
        test_folders (str): folders containing the test images.
        min_size (int): the minimum size of the images to accept.
        output_folder (str): the name of the folder where the 2 created files will be.
    """

    train_file = list()
    test_file = list()
    folders = [train_folders, test_folders]
    files = [train_file, test_file]
    files_name = ["train_images", "test_images"]

    for i in range(2):
        for folder in folders[i]:
            for file in os.listdir(folder):
                image_path = os.path.join(folder, file)
                image = Image.open(image_path, mode='r')
                if image.width >= min_size and image.height >= min_size:
                    files[i].append(image_path)

        print(f"There are {len(files[i])} images in the {files_name[i]}.\n")
        with open(os.path.join(output_folder, f"{files_name[i]}.json"), 'w') as f:
            json.dump(files[i], f)

    print("Finish ! The 2 JSONS of Train and Test images have been saved to %s\n" % output_folder)


# IF we do not create the LR images ourselves
def prepare_datasets_HRLR(train_HR_folders, train_LR_folders, test_folders, min_size, output_folder):
    """Creates 3 JSON files containing the path for the train and test images.
       (1. train_HR_images.json
        2. train_LR_images.json
        3. test_images.json)

    Args:
        train_HR_folders (str): folders containing the HR training images.
        train_LR_folders (str): folders containing the LR training images.
        test_folders (str): folders containing the test images.
        min_size (int): the minimum size of the images to accept.
        output_folder (str): the name of the folder where the 2 created files will be.
    """

    train_HR_file = list()
    train_LR_file = list()
    test_file = list()

    folders = [train_HR_folders, train_LR_folders, test_folders]
    files = [train_HR_file, train_LR_file, test_file]
    files_name = ["train_HR_images", "train_LR_images", "test_images"]

    for i in range(3):
        for folder in folders[i]:
            for file in os.listdir(folder):
                image_path = os.path.join(folder, file)
                image = Image.open(image_path, mode='r')
                if image.width >= min_size and image.height >= min_size:
                    files[i].append(image_path)

        print(f"There are {len(files[i])} images in the {files_name[i]}.\n")
        with open(os.path.join(output_folder, f"{files_name[i]}.json"), 'w') as f:
            json.dump(files[i], f)

    print("Finish ! The 3 JSONS of Train and Test images have been saved to %s\n" % output_folder)


if __name__ == '__main__':
    prepare_datasets(train_folders=['../data/DIV2K_train_HR'],
                      test_folders=['../data/test'],
                      min_size=100,
                      output_folder='./')

