from PIL import Image
# from numpy import asarray
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize, Compose
from torch import zeros
# import os

class MicroImages(Dataset):
    def __init__(self, root_dir, resize_size=48, dataset_size=800):
        """Initializes a dataset containing images and labels."""
        super().__init__()
        self.root_dir = root_dir
        self.transform_lr = Compose([ToTensor(), Resize((resize_size, resize_size))])
        self.transform_hr = Compose([ToTensor(), Resize((2*resize_size, 2*resize_size))])
        # self.transform_hr = Compose([ToTensor(), torch.nn.ZeroPad2d(2*resize_size),
        #                         transforms.CenterCrop(2*resize_size)])

        self.data = []
        for i in range(min(dataset_size, 800)):
            image_hr_dir = self.root_dir + 'DIV2K_train_HR/' + "{:0>4}".format(i+1) + '.png'
            image_lr_dir = self.root_dir + 'DIV2K_train_LR_bicubic/X2/' + "{:0>4}x2".format(i+1) + '.png'
            image_hr = Image.open(image_hr_dir)
            image_lr = Image.open(image_lr_dir)
            self.data.append((self.transform_lr(image_lr), self.transform_hr(image_hr)))

        self.size=len(self.data)

    def __len__(self):
        """Returns the size of the dataset."""
        return len(self.data)

    def __getitem__(self, index):
        """Returns the index-th data item of the dataset."""

        return self.data[index]