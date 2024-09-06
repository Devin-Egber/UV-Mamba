from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np


class UVSegDataset(Dataset):
    def __init__(self, img_path, mode="train", transform=None):
        """
        The dataset organization should have the following attributes:

        ├── dataset
        │   ├── img_dir
        │   │   ├── train
        │   │   ├── val
        │   │   ├── test
        │   ├── ann_dir
        │   │   ├── train
        │   │   ├── val
        │   │   ├── test

        """
        super(UVSegDataset, self).__init__()
        self.folder = img_path

        # Get metadata
        image_dir = os.path.join(img_path, "img_dir/{}".format(mode))
        mask_dir = os.path.join(img_path, "ann_dir/{}".format(mode))
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])

        img_name = img_path.split("/")[-1][0:-4]
        mask_name = mask_path.split("/")[-1][0:-4]
        assert img_name == mask_name

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path), dtype=np.uint8)
        mask[mask != 0] = 1

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']

        # return image, mask, img_name
        return image, mask