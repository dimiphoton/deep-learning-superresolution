from PIL import Image
# from numpy import asarray
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch import zeros
# import os

class UpscaledImages(Dataset):
    def __init__(self, root_dir, transform_lr, transform_hr, train=True):
        """Initializes a dataset containing images and labels."""
        super().__init__()
        self.root_dir = root_dir
        self.transform_lr = transform_lr
        self.transform_hr = transform_hr
        self.train=train
        self.size=800

    def __len__(self):
        """Returns the size of the dataset."""
        return self.size

    def __getitem__(self, index):
        """Returns the index-th data item of the dataset."""

        # if index < 0 or index >= self.size:
        #     raise ValueError("Wrong index value")

        if self.train:
                image_hr_dir = self.root_dir + 'DIV2K_train_HR/' + "{:0>4}".format(index+1) + '.png'
                image_lr_dir = self.root_dir + 'DIV2K_train_LR_bicubic/X2/' + "{:0>4}x2".format(index+1) + '.png'

        else: # test
            raise ValueError("No test set available")
        try:
            image_hr = Image.open(image_hr_dir)
            image_lr = Image.open(image_lr_dir)
            res = self.transform_lr(image_lr), self.transform_hr(image_hr)
            return res
        except Exception as err:
            print(err)
            print(f'Error in extracting image {index + 1} in {"train" if self.train else "test"} set')
            return zeros((3, 32, 32)), zeros((3, 64, 64))